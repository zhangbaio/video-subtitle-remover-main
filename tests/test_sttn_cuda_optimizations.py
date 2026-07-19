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


if __name__ == "__main__":
    unittest.main()
