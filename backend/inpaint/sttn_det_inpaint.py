import time
from contextlib import contextmanager

import cv2
import numpy as np
import torch
from torchvision import transforms
from typing import List, Union
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from backend.config import config
from backend.inpaint.sttn.network_sttn import InpaintGenerator
from backend.inpaint.utils.sttn_utils import Stack, ToTorchFormatTensor
from backend.tools.inpaint_tools import get_inpaint_area_by_mask


def _configure_cuda_inference(device):
    """Enable safe float32 CUDA inference optimizations for fixed-size STTN input."""
    try:
        device_type = torch.device(device).type
    except (TypeError, RuntimeError):
        device_type = getattr(device, "type", None)

    if device_type != "cuda":
        return False

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    set_matmul_precision = getattr(torch, "set_float32_matmul_precision", None)
    if callable(set_matmul_precision):
        set_matmul_precision("high")
    return True


@contextmanager
def _cuda_inference_optimizations(device):
    """Temporarily enable CUDA fast paths without leaking them to other models."""
    try:
        device_type = torch.device(device).type
    except (TypeError, RuntimeError):
        device_type = getattr(device, "type", None)

    if device_type != "cuda":
        yield
        return

    original_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    original_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    original_cudnn_benchmark = torch.backends.cudnn.benchmark
    get_matmul_precision = getattr(torch, "get_float32_matmul_precision", None)
    original_matmul_precision = (
        get_matmul_precision() if callable(get_matmul_precision) else None
    )
    _configure_cuda_inference(device)
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = original_matmul_tf32
        torch.backends.cudnn.allow_tf32 = original_cudnn_tf32
        torch.backends.cudnn.benchmark = original_cudnn_benchmark
        set_matmul_precision = getattr(torch, "set_float32_matmul_precision", None)
        if callable(set_matmul_precision) and original_matmul_precision is not None:
            set_matmul_precision(original_matmul_precision)


# 定义图像预处理方式
_to_tensors = transforms.Compose([
    Stack(),  # 将图像堆叠为序列
    ToTorchFormatTensor()  # 将堆叠的图像转化为PyTorch张量
])

class STTNDetInpaint:
    def __init__(self, device, model_path):
        self.device = device
        # 1. 创建InpaintGenerator模型实例并装载到选择的设备上
        self.model = InpaintGenerator().to(self.device)
        # 2. 载入预训练模型的权重，转载模型的状态字典
        self.model.load_state_dict(torch.load(model_path, map_location='cpu')['netG'])
        # 3. # 将模型设置为评估模式
        self.model.eval()
        # 模型输入用的宽和高
        self.model_input_width, self.model_input_height = 432, 240
        # 2. 设置相连帧数
        self.neighbor_stride = config.sttnNeighborStride.value
        self.ref_length = config.sttnReferenceLength.value

    def __call__(self, input_frames: List[np.ndarray], input_mask: np.ndarray):
        """
        :param input_frames: 原视频帧
        :param mask: 字幕区域mask
        """
        mask = input_mask[:, :, None]
        H_ori, W_ori = mask.shape[:2]
        H_ori = int(H_ori + 0.5)
        W_ori = int(W_ori + 0.5)
        # 确定去字幕的垂直高度部分
        if H_ori > W_ori:
            split_h = int(H_ori * 5 / 9)
        else:
            split_h = int(W_ori * 5 / 18)
        inpaint_area = get_inpaint_area_by_mask(W_ori, H_ori, split_h, mask)
        # 初始化帧存储变量
        # Keep caller-owned high-resolution frames read-only during inference.  A
        # frame is copied only when its final composited result is produced, so
        # we do not hold an otherwise unused full-resolution duplicate of the
        # whole batch while STTN is running.
        frames_hr = input_frames
        frames_scaled = {}  # 存放缩放后帧的字典
        masks_scaled = {}  # 存放缩放后遮罩的字典
        comps = {}  # 存放补全后帧的字典
        # 存储最终的视频帧
        inpainted_frames = []
        for k in range(len(inpaint_area)):
            frames_scaled[k] = []  # 为每个去除部分初始化一个列表
            masks_scaled[k] = None

            # The subtitle mask is static for every frame in this STTN batch.
            # Resize it once per crop instead of once per video frame.
            mask_crop = mask[inpaint_area[k][0]:inpaint_area[k][1], :, :]
            masks_scaled[k] = cv2.resize(
                mask_crop,
                (self.model_input_width, self.model_input_height),
            )

        # 读取并缩放帧
        for j in range(len(frames_hr)):
            image = frames_hr[j]
            # 对每个去除部分进行切割和缩放
            for k in range(len(inpaint_area)):
                image_crop = image[inpaint_area[k][0]:inpaint_area[k][1], :, :]  # 切割
                image_resize = cv2.resize(image_crop, (self.model_input_width, self.model_input_height))  # 缩放
                frames_scaled[k].append(image_resize)  # 将缩放后的帧添加到对应列表

        # 处理每一个去除部分
        for k in range(len(inpaint_area)):
            # 调用inpaint函数进行处理
            comps[k] = self.inpaint(frames_scaled[k], masks_scaled[k])

        # 如果存在去除部分
        if inpaint_area:
            for j in range(len(frames_hr)):
                frame = frames_hr[j].copy()
                # 对于模式中的每一个段落
                for k in range(len(inpaint_area)):
                    y0, y1 = inpaint_area[k][0], inpaint_area[k][1]
                    comp = cv2.resize(
                        comps[k][j],
                        (W_ori, y1 - y0),
                    ).astype(np.uint8, copy=False)
                    # ``inpaint`` returns OpenCV-native BGR frames. Copy only
                    # pixels covered by the original-resolution subtitle mask;
                    # replacing the complete STTN crop band would also replace
                    # untouched pixels after a lossy down/up-scale round trip.
                    crop = frame[y0:y1, :, :]
                    mask_area = mask[y0:y1, :, :] > 0
                    np.copyto(crop, comp, where=mask_area)
                # 将最终帧添加到列表
                inpainted_frames.append(frame)
                # print(f'processing frame, {len(frames_hr) - j} left')
        else:
            # Preserve the previous non-mutating API even when the mask contains
            # no valid inpaint area.
            inpainted_frames = [frame.copy() for frame in frames_hr]
        return inpainted_frames

    @staticmethod
    def read_mask(path):
        img = cv2.imread(path, 0)
        # 转为binary mask
        ret, img = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)
        img = img[:, :, None]
        return img

    def get_ref_index(self, neighbor_ids, length):
        """
        采样整个视频的参考帧
        """
        # 初始化参考帧的索引列表
        ref_index = []
        # 在视频长度范围内根据ref_length逐步迭代
        for i in range(0, length, self.ref_length):
            # 如果当前帧不在近邻帧中
            if i not in neighbor_ids:
                # 将它添加到参考帧列表
                ref_index.append(i)
        # 返回参考帧索引列表
        return ref_index

    def inpaint(
        self,
        frames: List[np.ndarray],
        masks: Union[np.ndarray, List[np.ndarray]],
    ):
        """
        使用STTN完成空洞填充（空洞即被遮罩的区域）
        """
        frame_length = len(frames)
        # 对帧进行预处理转换为张量，并进行归一化
        # Stack converts list entries to PIL images in-place; pass a shallow list
        # copy so the original NumPy frames remain available for compositing.
        feats = _to_tensors(list(frames)).unsqueeze(0) * 2 - 1

        # The detector path uses one static mask for the complete frame batch.
        # Keep support for a per-frame mask list, but avoid materialising and
        # transferring N identical masks in the common static-mask case.
        static_mask = isinstance(masks, np.ndarray) or len(masks) == 1
        if static_mask:
            mask_array = np.asarray(masks if isinstance(masks, np.ndarray) else masks[0])
            if mask_array.ndim == 3 and mask_array.shape[-1] == 1:
                mask_array = mask_array[:, :, 0]
            mask_arrays = [mask_array]
            binary_masks = [
                np.expand_dims((mask_array > 0.5).astype(np.uint8), 2)
            ]
            masks_tensor = (
                (_to_tensors(mask_arrays).unsqueeze(0) > 0.5)
                .float()
                .to(self.device)
                .expand(1, frame_length, -1, -1, -1)
            )
        else:
            if len(masks) != frame_length:
                raise ValueError(
                    "STTN masks must contain either one static mask or one mask per frame"
                )
            mask_arrays = []
            for current_mask in masks:
                mask_array = np.asarray(current_mask)
                if mask_array.ndim == 3 and mask_array.shape[-1] == 1:
                    mask_array = mask_array[:, :, 0]
                mask_arrays.append(mask_array)
            binary_masks = [
                np.expand_dims((mask_array > 0.5).astype(np.uint8), 2)
                for mask_array in mask_arrays
            ]
            masks_tensor = (
                (_to_tensors(mask_arrays).unsqueeze(0) > 0.5)
                .float()
                .to(self.device)
            )

        # Move the frame tensor to the selected inference device.
        feats = feats.to(self.device)
        prediction_counts = [0] * frame_length
        # 统一关闭梯度计算，用于推理阶段节省内存并加速
        with _cuda_inference_optimizations(self.device), torch.inference_mode():
            # 将处理好的帧通过编码器，产生特征表示
            feats = self.model.encoder((feats*(1-masks_tensor).float()).view(frame_length, 3, self.model_input_height, self.model_input_width))
            # 获取特征维度信息
            _, c, feat_h, feat_w = feats.size()
            # 调整特征形状以匹配模型的期望输入
            feats = feats.view(1, frame_length, c, feat_h, feat_w)
            # Accumulate overlapping-window predictions on the inference device.
            # The previous implementation copied every window to the CPU; this
            # buffer lets the whole batch use a single device-to-host transfer.
            prediction_accumulator = torch.empty(
                (
                    frame_length,
                    3,
                    self.model_input_height,
                    self.model_input_width,
                ),
                dtype=torch.float32,
                device=self.device,
            )
            # 在设定的邻居帧步幅内循环处理视频
            for f in range(0, frame_length, self.neighbor_stride):
                # 计算邻近帧的ID
                neighbor_ids = [i for i in range(max(0, f - self.neighbor_stride), min(frame_length, f + self.neighbor_stride + 1))]
                # 获取参考帧的索引
                ref_ids = self.get_ref_index(neighbor_ids, frame_length)
                # 通过模型推断特征并传递给解码器以生成完成的帧
                pred_feat = self.model.infer(
                    feats[0, neighbor_ids + ref_ids, :, :, :], masks_tensor[0, neighbor_ids + ref_ids, :, :, :])

                # 将预测的特征通过解码器生成图片，并应用激活函数tanh
                pred_img = torch.tanh(self.model.decoder(pred_feat[:len(neighbor_ids), :, :, :]))
                # 将结果张量重新缩放到0到255的范围内（图像像素值）
                pred_img = (pred_img + 1) / 2
                # Match NumPy's previous uint8 truncation before merging
                # overlapping predictions, while keeping the merge on-device.
                pred_img = pred_img.mul(255).clamp_(0, 255).floor_().float()
                # 遍历邻近帧
                for i in range(len(neighbor_ids)):
                    idx = neighbor_ids[i]
                    if prediction_counts[idx] == 0:
                        prediction_accumulator[idx].copy_(pred_img[i])
                    else:
                        prediction_accumulator[idx].mul_(0.5).add_(pred_img[i], alpha=0.5)
                    prediction_counts[idx] += 1

            # Exactly one D2H transfer for all completed frames in this crop.
            predictions_cpu = prediction_accumulator.cpu().numpy()

        comp_frames = [None] * frame_length
        for idx, prediction_count in enumerate(prediction_counts):
            if prediction_count == 0:
                continue
            # STTN predicts RGB tensors, while every OpenCV frame in ``frames``
            # is BGR. Convert before compositing so the two colour spaces can
            # never be mixed into a blue/purple full-width crop band.
            prediction = predictions_cpu[idx].transpose(1, 2, 0)[..., ::-1]
            binary_mask = binary_masks[0 if static_mask else idx]
            if prediction_count == 1:
                # Preserve the old single-prediction uint8 result type.
                prediction = prediction.astype(np.uint8, copy=False)
                comp_frames[idx] = (
                    prediction * binary_mask
                    + frames[idx] * (1 - binary_mask)
                )
            else:
                # Repeated 50/50 merging produces float32 in the original path.
                comp_frames[idx] = (
                    prediction * binary_mask
                    + frames[idx].astype(np.float32) * (1 - binary_mask)
                )
        # 返回处理完成的帧序列
        return comp_frames
