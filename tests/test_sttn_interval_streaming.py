import unittest
from unittest import mock

import numpy as np

from backend.main import SubtitleRemover


class _FakeCapture:
    def __init__(self, frames):
        self.frames = list(frames)
        self.index = 0

    def read(self):
        if self.index >= len(self.frames):
            return False, None
        frame = self.frames[self.index]
        self.index += 1
        return True, frame


class _SyncReader:
    def __init__(self, capture):
        self.capture = capture
        self.stopped = False

    def read(self):
        return self.capture.read()

    def stop(self):
        self.stopped = True


class _CollectingWriter:
    def __init__(self):
        self.frames = []

    def write(self, frame):
        self.frames.append(frame.copy())


class _FakeProgressBar:
    def __init__(self, total):
        self.total = total
        self.n = 0
        self.increments = []
        self.messages = []

    def update(self, increment):
        self.increments.append(increment)
        self.n += increment

    def write(self, message):
        self.messages.append(message)


def _numbered_frames(count):
    return [
        np.full((2, 2, 3), frame_no, dtype=np.uint8)
        for frame_no in range(1, count + 1)
    ]


class SttnIntervalStreamingTests(unittest.TestCase):
    def _run_video_inpaint(
        self,
        *,
        actual_frame_count,
        declared_frame_count,
        interval,
        max_load_num,
        final_frame_area=None,
        model_exception=None,
    ):
        frames = _numbered_frames(actual_frame_count)
        capture = _FakeCapture(frames)
        reader = _SyncReader(capture)
        writer = _CollectingWriter()
        progress = _FakeProgressBar(total=declared_frame_count)
        previews = []
        mask = np.zeros((2, 2), dtype=np.uint8)
        area = (0, 1, 0, 1)
        start, end = interval
        timeline = {
            frame_no: [area]
            for frame_no in range(start, end + 1)
        }
        if final_frame_area is not None:
            timeline[end] = [final_frame_area]

        detector = mock.Mock()
        detector.get_cached_timeline.return_value = timeline
        detector.find_continuous_ranges_with_same_mask.return_value = [interval]
        detector.filter_and_merge_intervals.side_effect = lambda ranges, _length: ranges

        remover = object.__new__(SubtitleRemover)
        remover.video_path = "video.mp4"
        remover.sub_areas = [(0, 2, 0, 2)]
        remover.subtitle_detection_cache = object()
        remover.ab_sections = None
        remover.frame_count = declared_frame_count
        remover.mask_size = mask.shape
        remover.video_cap = capture
        remover.video_writer = writer
        remover.progress_total = 0
        remover.progress_remover = 0
        remover.append_output = mock.Mock()
        remover.notify_progress_listeners = mock.Mock()
        remover.push_preview_with_comp = (
            lambda original, composed: previews.append(
                (original.copy(), composed.copy())
            )
        )

        model_batches = []
        model_masks = []

        def model(batch, batch_mask):
            model_batches.append([int(frame[0, 0, 0]) for frame in batch])
            model_masks.append(batch_mask)
            if model_exception is not None:
                raise model_exception
            return [frame + 100 for frame in batch]

        with (
            mock.patch("backend.main.SubtitleDetect", return_value=detector),
            mock.patch("backend.main.FramePrefetcher", return_value=reader),
            mock.patch(
                "backend.main.expand_frame_ranges",
                side_effect=lambda ranges, _backward, _forward: ranges,
            ),
            mock.patch("backend.main.create_mask", return_value=mask) as create_mask,
            mock.patch(
                "backend.main.config.getSttnMaxLoadNum",
                return_value=max_load_num,
            ),
        ):
            remover.video_inpaint(progress, model)

        return {
            "create_mask": create_mask,
            "mask": mask,
            "model_batches": model_batches,
            "model_masks": model_masks,
            "previews": previews,
            "progress": progress,
            "reader": reader,
            "writer": writer,
        }

    def test_long_interval_streams_in_bounded_ordered_batches(self):
        result = self._run_video_inpaint(
            actual_frame_count=16,
            declared_frame_count=16,
            interval=(2, 14),
            max_load_num=5,
        )

        self.assertEqual(
            result["model_batches"],
            [list(range(2, 7)), list(range(7, 12)), list(range(12, 15))],
        )
        self.assertTrue(
            all(len(batch) <= 5 for batch in result["model_batches"])
        )
        self.assertTrue(
            all(batch_mask is result["mask"] for batch_mask in result["model_masks"])
        )
        result["create_mask"].assert_called_once_with(
            (2, 2),
            [(0, 1, 0, 1)],
        )

        written_numbers = [
            int(frame[0, 0, 0]) for frame in result["writer"].frames
        ]
        self.assertEqual(
            written_numbers,
            [1] + list(range(102, 115)) + [15, 16],
        )
        self.assertEqual(result["progress"].increments, [1, 5, 5, 3, 1, 1])
        self.assertEqual(result["progress"].n, 16)
        self.assertEqual(len(result["previews"]), 16)
        self.assertTrue(result["reader"].stopped)

    def test_streaming_preserves_existing_balanced_batch_lengths(self):
        result = self._run_video_inpaint(
            actual_frame_count=50,
            declared_frame_count=50,
            interval=(1, 50),
            max_load_num=50,
        )

        # batch_generator historically balances 50 frames as 33 + 17. The
        # streaming implementation derives these lengths without retaining
        # the corresponding frame arrays.
        self.assertEqual(
            [len(batch) for batch in result["model_batches"]],
            [33, 17],
        )
        self.assertEqual(result["progress"].increments, [33, 17])
        self.assertEqual(result["progress"].n, 50)

    def test_interval_mask_includes_a_box_unique_to_the_final_frame(self):
        final_area = (1, 2, 1, 2)
        result = self._run_video_inpaint(
            actual_frame_count=6,
            declared_frame_count=6,
            interval=(2, 5),
            max_load_num=4,
            final_frame_area=final_area,
        )

        result["create_mask"].assert_called_once_with(
            (2, 2),
            [(0, 1, 0, 1), final_area],
        )

    def test_early_eof_processes_partial_batch_and_returns(self):
        result = self._run_video_inpaint(
            actual_frame_count=7,
            declared_frame_count=10,
            interval=(1, 10),
            max_load_num=4,
        )

        self.assertEqual(
            result["model_batches"],
            [list(range(1, 5)), list(range(5, 8))],
        )
        self.assertEqual(len(result["writer"].frames), 7)
        self.assertEqual(result["progress"].n, 7)
        self.assertTrue(result["reader"].stopped)

    def test_model_failure_still_stops_frame_prefetcher(self):
        with (
            mock.patch.object(
                _SyncReader,
                "stop",
                autospec=True,
            ) as stop_reader,
            self.assertRaisesRegex(RuntimeError, "inference failed"),
        ):
            self._run_video_inpaint(
                actual_frame_count=8,
                declared_frame_count=8,
                interval=(1, 8),
                max_load_num=4,
                model_exception=RuntimeError("inference failed"),
            )

        stop_reader.assert_called_once()


if __name__ == "__main__":
    unittest.main()
