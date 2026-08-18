import unittest
from unittest import mock

import numpy as np
import torch

from backend.inpaint import sttn_det_inpaint
from backend.inpaint.sttn.network_sttn import Attention


class SttnAttentionMaskTests(unittest.TestCase):
    def test_attention_excludes_masked_keys(self):
        attention = Attention()
        query = torch.ones((1, 2, 1), dtype=torch.float32)
        key = torch.ones((1, 2, 1), dtype=torch.float32)
        value = torch.tensor([[[1.0], [9.0]]])
        mask = torch.tensor(
            [[[False, True], [False, True]]], dtype=torch.bool
        )

        output, weights = attention(query, key, value, mask)

        self.assertTrue(torch.allclose(output, torch.ones_like(output)))
        self.assertTrue(torch.equal(weights[..., 1], torch.zeros_like(weights[..., 1])))


class SttnCudaConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.original_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
        self.original_cudnn_tf32 = torch.backends.cudnn.allow_tf32
        self.original_benchmark = torch.backends.cudnn.benchmark
        self.original_matmul_precision = torch.get_float32_matmul_precision()

    def tearDown(self):
        torch.backends.cuda.matmul.allow_tf32 = self.original_matmul_tf32
        torch.backends.cudnn.allow_tf32 = self.original_cudnn_tf32
        torch.backends.cudnn.benchmark = self.original_benchmark
        torch.set_float32_matmul_precision(self.original_matmul_precision)

    def test_cuda_enables_float32_inference_optimizations(self):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False

        with mock.patch.object(torch, "set_float32_matmul_precision") as set_precision:
            enabled = sttn_det_inpaint._configure_cuda_inference("cuda:0")

        self.assertTrue(enabled)
        self.assertTrue(torch.backends.cuda.matmul.allow_tf32)
        self.assertTrue(torch.backends.cudnn.allow_tf32)
        self.assertTrue(torch.backends.cudnn.benchmark)
        set_precision.assert_called_once_with("high")

    def test_cpu_does_not_change_cuda_configuration(self):
        expected = (
            torch.backends.cuda.matmul.allow_tf32,
            torch.backends.cudnn.allow_tf32,
            torch.backends.cudnn.benchmark,
        )

        with mock.patch.object(torch, "set_float32_matmul_precision") as set_precision:
            enabled = sttn_det_inpaint._configure_cuda_inference("cpu")

        self.assertFalse(enabled)
        self.assertEqual(
            expected,
            (
                torch.backends.cuda.matmul.allow_tf32,
                torch.backends.cudnn.allow_tf32,
                torch.backends.cudnn.benchmark,
            ),
        )
        set_precision.assert_not_called()

    def test_cuda_context_restores_process_wide_settings(self):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        original_precision = torch.get_float32_matmul_precision()

        with sttn_det_inpaint._cuda_inference_optimizations("cuda:0"):
            self.assertTrue(torch.backends.cuda.matmul.allow_tf32)
            self.assertTrue(torch.backends.cudnn.allow_tf32)
            self.assertTrue(torch.backends.cudnn.benchmark)
            self.assertEqual(torch.get_float32_matmul_precision(), "high")

        self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
        self.assertFalse(torch.backends.cudnn.allow_tf32)
        self.assertFalse(torch.backends.cudnn.benchmark)
        self.assertEqual(torch.get_float32_matmul_precision(), original_precision)


class _InferenceModeProbeModel:
    def __init__(self):
        self.inference_mode_states = []

    def _record(self):
        self.inference_mode_states.append(torch.is_inference_mode_enabled())

    def encoder(self, frames):
        self._record()
        return frames

    def infer(self, features, _masks):
        self._record()
        return features

    def decoder(self, features):
        self._record()
        return features


class SttnInferenceModeTests(unittest.TestCase):
    def test_inpaint_model_calls_run_in_inference_mode_on_cpu(self):
        inpainter = object.__new__(sttn_det_inpaint.STTNDetInpaint)
        inpainter.device = torch.device("cpu")
        inpainter.model_input_width = 4
        inpainter.model_input_height = 4
        inpainter.neighbor_stride = 1
        inpainter.ref_length = 1
        inpainter.model = _InferenceModeProbeModel()
        frames = [np.zeros((4, 4, 3), dtype=np.uint8)]
        masks = [np.zeros((4, 4), dtype=np.uint8)]

        result = inpainter.inpaint(frames, masks)

        self.assertEqual(len(result), 1)
        self.assertTrue(inpainter.model.inference_mode_states)
        self.assertTrue(all(inpainter.model.inference_mode_states))


class _WindowVaryingProbeModel:
    def __init__(self, values):
        self.values = values
        self.infer_calls = 0

    def encoder(self, frames):
        return frames

    def infer(self, features, _masks):
        value = self.values[self.infer_calls]
        self.infer_calls += 1
        return torch.full_like(features, value)

    def decoder(self, features):
        return features


class _RgbChannelProbeModel:
    def encoder(self, frames):
        return frames

    def infer(self, features, _masks):
        return features

    def decoder(self, features):
        rgb = torch.tensor(
            [-0.8, 0.0, 0.8],
            dtype=features.dtype,
            device=features.device,
        ).view(1, 3, 1, 1)
        return rgb.expand(
            features.shape[0],
            3,
            features.shape[2],
            features.shape[3],
        )


class SttnBatchTransferTests(unittest.TestCase):
    @staticmethod
    def _make_inpainter(model, frame_size=4):
        inpainter = object.__new__(sttn_det_inpaint.STTNDetInpaint)
        inpainter.device = torch.device("cpu")
        inpainter.model_input_width = frame_size
        inpainter.model_input_height = frame_size
        inpainter.neighbor_stride = 2
        inpainter.ref_length = 2
        inpainter.model = model
        return inpainter

    def test_device_side_overlap_merge_matches_previous_cpu_semantics(self):
        values = [-0.75, 0.10, 0.80]
        inpainter = self._make_inpainter(_WindowVaryingProbeModel(values))
        rng = np.random.default_rng(1234)
        frames = [
            rng.integers(0, 256, (4, 4, 3), dtype=np.uint8)
            for _ in range(5)
        ]
        original_frames = [frame.copy() for frame in frames]
        mask = np.zeros((4, 4), dtype=np.uint8)
        mask[1:3, 1:3] = 1

        actual = inpainter.inpaint(frames, mask)

        expected = [None] * len(frames)
        binary_mask = mask[:, :, None]
        for call_index, f in enumerate(range(0, len(frames), 2)):
            neighbor_ids = list(
                range(max(0, f - 2), min(len(frames), f + 3))
            )
            prediction = torch.tanh(torch.tensor(values[call_index]))
            pixel = np.uint8(float((prediction + 1) / 2 * 255))
            for idx in neighbor_ids:
                image = (
                    np.full_like(frames[idx], pixel) * binary_mask
                    + frames[idx] * (1 - binary_mask)
                )
                if expected[idx] is None:
                    expected[idx] = image
                else:
                    expected[idx] = (
                        expected[idx].astype(np.float32) * 0.5
                        + image.astype(np.float32) * 0.5
                    )

        for expected_frame, actual_frame in zip(expected, actual):
            np.testing.assert_array_equal(expected_frame, actual_frame)
        for original, current in zip(original_frames, frames):
            np.testing.assert_array_equal(original, current)

    def test_static_mask_is_resized_once_per_crop(self):
        inpainter = self._make_inpainter(_InferenceModeProbeModel(), frame_size=8)
        inpainter.model_input_height = 4
        frames = [
            np.full((8, 8, 3), value, dtype=np.uint8)
            for value in (10, 20, 30)
        ]
        originals = [frame.copy() for frame in frames]
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[4:6, 2:6] = 1
        fake_comps = [np.zeros((4, 8, 3), dtype=np.uint8) for _ in frames]

        real_resize = sttn_det_inpaint.cv2.resize
        with (
            mock.patch.object(
                sttn_det_inpaint,
                "get_inpaint_area_by_mask",
                return_value=[(4, 6, 0, 8)],
            ),
            mock.patch.object(inpainter, "inpaint", return_value=fake_comps) as inpaint,
            mock.patch.object(
                sttn_det_inpaint.cv2,
                "resize",
                wraps=real_resize,
            ) as resize,
        ):
            result = inpainter(frames, mask)

        resized_masks = [
            call
            for call in resize.call_args_list
            if call.args[0].ndim == 3 and call.args[0].shape[-1] == 1
        ]
        self.assertEqual(len(resized_masks), 1)
        static_mask = inpaint.call_args.args[1]
        self.assertIsInstance(static_mask, np.ndarray)
        self.assertEqual(static_mask.shape, (4, 8))
        for original, current, output in zip(originals, frames, result):
            np.testing.assert_array_equal(original, current)
            self.assertFalse(np.shares_memory(output, current))

    def test_model_rgb_prediction_is_returned_as_bgr(self):
        inpainter = self._make_inpainter(_RgbChannelProbeModel())
        frames = [np.zeros((4, 4, 3), dtype=np.uint8)]
        mask = np.ones((4, 4), dtype=np.uint8)

        result = inpainter.inpaint(frames, mask)

        rgb = (
            (torch.tanh(torch.tensor([-0.8, 0.0, 0.8])) + 1)
            / 2
            * 255
        ).floor().to(torch.uint8).numpy()
        expected_bgr = rgb[::-1]
        np.testing.assert_array_equal(result[0][0, 0], expected_bgr)

    def test_full_resolution_composite_changes_only_mask_pixels(self):
        inpainter = self._make_inpainter(_InferenceModeProbeModel(), frame_size=8)
        inpainter.model_input_height = 4
        frame = np.empty((8, 8, 3), dtype=np.uint8)
        frame[:, :] = [10, 20, 200]
        original = frame.copy()
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[4:6, 2:6] = 1
        predicted_bgr = np.array([210, 40, 15], dtype=np.uint8)
        fake_comp = np.empty((4, 8, 3), dtype=np.uint8)
        fake_comp[:, :] = predicted_bgr

        with (
            mock.patch.object(
                sttn_det_inpaint,
                "get_inpaint_area_by_mask",
                return_value=[(4, 6, 0, 8)],
            ),
            mock.patch.object(inpainter, "inpaint", return_value=[fake_comp]),
        ):
            result = inpainter([frame], mask)[0]

        outside_mask = mask == 0
        np.testing.assert_array_equal(result[outside_mask], original[outside_mask])
        np.testing.assert_array_equal(
            result[mask == 1],
            np.broadcast_to(predicted_bgr, result[mask == 1].shape),
        )
        np.testing.assert_array_equal(frame, original)

    def test_empty_mask_returns_independent_frame_copies(self):
        inpainter = self._make_inpainter(_InferenceModeProbeModel())
        frames = [np.arange(48, dtype=np.uint8).reshape(4, 4, 3)]
        result = inpainter(frames, np.zeros((4, 4), dtype=np.uint8))

        np.testing.assert_array_equal(result[0], frames[0])
        self.assertFalse(np.shares_memory(result[0], frames[0]))


if __name__ == "__main__":
    unittest.main()
