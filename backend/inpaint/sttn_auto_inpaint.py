import gc
import os
import sys
import time
from typing import List

import cv2
import numpy as np
import torch
from tqdm import tqdm
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.config import config
from backend.inpaint.sttn.auto_sttn import InpaintGenerator
from backend.inpaint.utils.sttn_utils import Stack, ToTorchFormatTensor
from backend.tools.hardware_accelerator import HardwareAccelerator
from backend.tools.inpaint_tools import get_inpaint_area_by_mask, is_frame_number_in_ab_sections
from backend.tools.video_io import FramePrefetcher


_to_tensors = transforms.Compose([
    Stack(),
    ToTorchFormatTensor(),
])


class STTNInpaint:
    def __init__(self, device, model_path):
        self.device = device
        self.model = InpaintGenerator().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location="cpu")["netG"])
        self.model.eval()
        self.model_input_width, self.model_input_height = 640, 120
        self.neighbor_stride = config.sttnNeighborStride.value
        self.ref_length = config.sttnReferenceLength.value

    def __call__(self, input_frames: List[np.ndarray], input_mask: np.ndarray):
        _, mask = cv2.threshold(input_mask, 127, 1, cv2.THRESH_BINARY)
        mask = mask[:, :, None]
        h_ori, w_ori = mask.shape[:2]
        split_h = int(w_ori * 3 / 16)
        inpaint_area = get_inpaint_area_by_mask(w_ori, h_ori, split_h, mask)

        frames_hr = [frame.copy() for frame in input_frames]
        frames_scaled = {k: [] for k in range(len(inpaint_area))}
        comps = {}

        for image in frames_hr:
            for k in range(len(inpaint_area)):
                image_crop = image[inpaint_area[k][0]:inpaint_area[k][1], :, :]
                image_resize = cv2.resize(
                    image_crop,
                    (self.model_input_width, self.model_input_height),
                )
                frames_scaled[k].append(image_resize)

        for k in range(len(inpaint_area)):
            comps[k] = self.inpaint(frames_scaled[k])

        if not inpaint_area:
            return frames_hr

        inpainted_frames = []
        for j, frame in enumerate(frames_hr):
            for k in range(len(inpaint_area)):
                comp = cv2.resize(comps[k][j], (w_ori, split_h))
                comp = cv2.cvtColor(comp.astype(np.uint8), cv2.COLOR_BGR2RGB)
                mask_area = mask[inpaint_area[k][0]:inpaint_area[k][1], :]
                frame[inpaint_area[k][0]:inpaint_area[k][1], :, :] = (
                    mask_area * comp
                    + (1 - mask_area) * frame[inpaint_area[k][0]:inpaint_area[k][1], :, :]
                )
            inpainted_frames.append(frame)

        return inpainted_frames

    @staticmethod
    def read_mask(path):
        img = cv2.imread(path, 0)
        _, img = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)
        return img[:, :, None]

    def get_ref_index(self, neighbor_ids, length):
        ref_index = []
        for i in range(0, length, self.ref_length):
            if i not in neighbor_ids:
                ref_index.append(i)
        return ref_index

    def inpaint(self, frames: List[np.ndarray]):
        frame_length = len(frames)
        feats = _to_tensors(frames).unsqueeze(0) * 2 - 1
        feats = feats.to(self.device)
        comp_frames = [None] * frame_length

        with torch.no_grad():
            feats = self.model.encoder(
                feats.view(frame_length, 3, self.model_input_height, self.model_input_width)
            )
            _, c, feat_h, feat_w = feats.size()
            feats = feats.view(1, frame_length, c, feat_h, feat_w)

            for f in range(0, frame_length, self.neighbor_stride):
                neighbor_ids = [
                    i for i in range(
                        max(0, f - self.neighbor_stride),
                        min(frame_length, f + self.neighbor_stride + 1),
                    )
                ]
                ref_ids = self.get_ref_index(neighbor_ids, frame_length)
                pred_feat = self.model.infer(feats[0, neighbor_ids + ref_ids, :, :, :])
                pred_img = torch.tanh(self.model.decoder(pred_feat[:len(neighbor_ids), :, :, :]))
                pred_img = (pred_img + 1) / 2
                pred_img = pred_img.cpu().permute(0, 2, 3, 1).numpy() * 255

                for i in range(len(neighbor_ids)):
                    idx = neighbor_ids[i]
                    img = pred_img[i].astype(np.uint8)
                    if comp_frames[idx] is None:
                        comp_frames[idx] = img
                    else:
                        comp_frames[idx] = (
                            comp_frames[idx].astype(np.float32) * 0.5
                            + img.astype(np.float32) * 0.5
                        )

        return comp_frames


class STTNAutoInpaint:
    def __init__(self, device, model_path, video_path, mask_path=None, clip_gap=None):
        self.sttn_inpaint = STTNInpaint(device, model_path)
        self.video_path = video_path
        self.mask_path = mask_path
        self.video_out_path = os.path.join(
            os.path.dirname(os.path.abspath(self.video_path)),
            f"{os.path.basename(self.video_path).rsplit('.', 1)[0]}_no_sub.mp4",
        )
        self.clip_gap = clip_gap if clip_gap is not None else config.getSttnMaxLoadNum()

    def read_frame_info_from_video(self):
        reader = cv2.VideoCapture(self.video_path)
        frame_info = {
            "W_ori": int(reader.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5),
            "H_ori": int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5),
            "fps": reader.get(cv2.CAP_PROP_FPS),
            "len": int(reader.get(cv2.CAP_PROP_FRAME_COUNT) + 0.5),
        }
        return reader, frame_info

    def __call__(self, input_mask=None, input_sub_remover=None, tbar=None):
        reader = None
        writer = None
        prefetcher = None

        try:
            reader, frame_info = self.read_frame_info_from_video()
            prefetcher = FramePrefetcher(reader)

            if input_sub_remover is not None:
                ab_sections = input_sub_remover.ab_sections
                writer = input_sub_remover.video_writer
            else:
                ab_sections = None
                writer = cv2.VideoWriter(
                    self.video_out_path,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    frame_info["fps"],
                    (frame_info["W_ori"], frame_info["H_ori"]),
                )

            split_h = int(frame_info["W_ori"] * 3 / 16)

            if input_mask is None:
                mask = self.sttn_inpaint.read_mask(self.mask_path)
            else:
                _, mask = cv2.threshold(input_mask, 127, 1, cv2.THRESH_BINARY)
                mask = mask[:, :, None]

            inpaint_area = get_inpaint_area_by_mask(
                frame_info["W_ori"],
                frame_info["H_ori"],
                split_h,
                mask,
            )

            effective_clip_gap = self.clip_gap
            vram_mb = HardwareAccelerator.instance().get_available_vram_mb()
            if vram_mb > 0:
                bytes_per_frame = frame_info["W_ori"] * frame_info["H_ori"] * 12
                max_frames_by_vram = int(vram_mb * 1024 * 1024 / max(bytes_per_frame, 1))
                max_frames_by_vram = max(max_frames_by_vram, 10)
                effective_clip_gap = min(self.clip_gap, max_frames_by_vram)
                if effective_clip_gap < self.clip_gap:
                    tqdm.write(
                        f"GPU VRAM: {vram_mb:.0f}MB, adjusting clip_gap: "
                        f"{self.clip_gap} -> {effective_clip_gap}"
                    )

            rec_time = (
                frame_info["len"] // effective_clip_gap
                if frame_info["len"] % effective_clip_gap == 0
                else frame_info["len"] // effective_clip_gap + 1
            )

            for i in range(rec_time):
                start_f = i * effective_clip_gap
                end_f = min((i + 1) * effective_clip_gap, frame_info["len"])
                tqdm.write(
                    f"Processing: {start_f + 1} - {end_f} / Total: {frame_info['len']}"
                )

                frames_hr = []
                frames = {k: [] for k in range(len(inpaint_area))}
                comps = {}

                valid_frames_count = 0
                for frame_no in range(start_f, end_f):
                    success, image = prefetcher.read()
                    if not success:
                        print(f"Warning: Failed to read frame {frame_no}.")
                        break

                    frames_hr.append(image)
                    valid_frames_count += 1

                    if is_frame_number_in_ab_sections(frame_no, ab_sections):
                        for k in range(len(inpaint_area)):
                            image_crop = image[inpaint_area[k][0]:inpaint_area[k][1], :, :]
                            image_resize = cv2.resize(
                                image_crop,
                                (
                                    self.sttn_inpaint.model_input_width,
                                    self.sttn_inpaint.model_input_height,
                                ),
                            )
                            frames[k].append(image_resize)

                if valid_frames_count == 0:
                    print(
                        f"Warning: No valid frames found in range "
                        f"{start_f + 1}-{end_f}. Skipping this segment."
                    )
                    continue

                for k in range(len(inpaint_area)):
                    comps[k] = (
                        self.sttn_inpaint.inpaint(frames[k])
                        if len(frames[k]) > 0
                        else []
                    )

                if inpaint_area and valid_frames_count > 0:
                    processed_frames_map = {}
                    processed_idx = 0

                    for local_idx, frame_no in enumerate(range(start_f, start_f + valid_frames_count)):
                        if is_frame_number_in_ab_sections(frame_no, ab_sections):
                            processed_frames_map[local_idx] = processed_idx
                            processed_idx += 1

                    for local_idx in range(valid_frames_count):
                        if input_sub_remover is not None and input_sub_remover.gui_mode:
                            original_frame = frames_hr[local_idx].copy()
                        else:
                            original_frame = None

                        frame = frames_hr[local_idx]

                        if local_idx in processed_frames_map:
                            comp_idx = processed_frames_map[local_idx]
                            for k in range(len(inpaint_area)):
                                if comp_idx < len(comps[k]):
                                    comp = cv2.resize(comps[k][comp_idx], (frame_info["W_ori"], split_h))
                                    comp = cv2.cvtColor(comp.astype(np.uint8), cv2.COLOR_BGR2RGB)
                                    mask_area = mask[inpaint_area[k][0]:inpaint_area[k][1], :]
                                    frame[inpaint_area[k][0]:inpaint_area[k][1], :, :] = (
                                        mask_area * comp
                                        + (1 - mask_area)
                                        * frame[inpaint_area[k][0]:inpaint_area[k][1], :, :]
                                    )

                        writer.write(frame)

                        if input_sub_remover is not None:
                            if tbar is not None:
                                input_sub_remover.update_progress(tbar, increment=1)
                            if original_frame is not None and input_sub_remover.gui_mode:
                                input_sub_remover.update_preview_with_comp(original_frame, frame)

                del frames_hr, frames, comps
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error during video processing: {str(e)}")
        finally:
            if prefetcher:
                prefetcher.release()
            elif reader:
                reader.release()
            if writer:
                writer.release()


if __name__ == "__main__":
    mask_path = "../../test/test.png"
    video_path = "../../test/test.mp4"
    start = time.time()
    accelerator = HardwareAccelerator.instance()
    sttn_video_inpaint = STTNAutoInpaint(
        accelerator.device,
        "../../backend/models/sttn-auto/infer_model.pth",
        video_path,
        mask_path=mask_path,
        clip_gap=config.getSttnMaxLoadNum(),
    )
    sttn_video_inpaint()
    print(f"video generated at {sttn_video_inpaint.video_out_path}")
    print(f"time cost: {time.time() - start}")
