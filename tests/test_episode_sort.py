from pathlib import Path
import os
import tempfile
import unittest

from backend.tools.episode_sort import chinese_number_to_int, episode_name_sort_key


def _sorted_names(names):
    return [
        Path(path).name
        for path in sorted((Path("videos") / name for name in names), key=episode_name_sort_key)
    ]


class EpisodeSortTests(unittest.TestCase):
    def test_chinese_number_to_int(self):
        cases = [
            ("一", 1),
            ("十", 10),
            ("十一", 11),
            ("二十一", 21),
            ("一百零二", 102),
            ("二〇二四", 2024),
            ("两千零六", 2006),
        ]
        for text, expected in cases:
            with self.subTest(text=text):
                self.assertEqual(chinese_number_to_int(text), expected)

    def test_sorts_arabic_episode_numbers_naturally(self):
        self.assertEqual(
            _sorted_names(["第10集.mp4", "第2集.mp4", "第1集.mp4"]),
            ["第1集.mp4", "第2集.mp4", "第10集.mp4"],
        )

    def test_sorts_chinese_episode_numbers_naturally(self):
        self.assertEqual(
            _sorted_names(["第二十一集.mp4", "第十集.mp4", "第二集.mp4", "第一集.mp4"]),
            ["第一集.mp4", "第二集.mp4", "第十集.mp4", "第二十一集.mp4"],
        )

    def test_normalizes_full_width_digits_and_is_deterministic(self):
        self.assertEqual(
            _sorted_names(["EP１０.mp4", "ep2.mp4", "ep０２.mp4", "ep1.mp4"]),
            ["ep1.mp4", "ep2.mp4", "ep０２.mp4", "EP１０.mp4"],
        )

    def test_uses_only_the_basename_for_folder_import_order(self):
        paths = [
            Path("z-folder") / "12.mp4",
            Path("a-folder") / "3.mp4",
            Path("m-folder") / "1.mp4",
        ]
        self.assertEqual(
            [path.name for path in sorted(paths, key=episode_name_sort_key)],
            ["1.mp4", "3.mp4", "12.mp4"],
        )

    def test_non_episode_names_use_case_insensitive_natural_fallback(self):
        self.assertEqual(
            _sorted_names(["Trailer10.mp4", "README.mp4", "trailer2.mp4", "bonus.mp4"]),
            ["bonus.mp4", "README.mp4", "trailer2.mp4", "Trailer10.mp4"],
        )

    def test_ignores_digits_in_the_file_extension(self):
        self.assertEqual(
            _sorted_names(["clip2.mp4", "clip.mp4", "clip10.mp4"]),
            ["clip.mp4", "clip2.mp4", "clip10.mp4"],
        )

    def test_folder_collection_filters_and_uses_natural_order(self):
        from ui.home_interface import HomeInterface

        with tempfile.TemporaryDirectory() as folder:
            for name in (
                "Episode10.MP4",
                "episode2.mkv",
                "special.mp4",
                "README.txt",
                "poster.png",
                "episode1_NO_SUB.mp4",
            ):
                (Path(folder) / name).write_bytes(b"")
            (Path(folder) / "episode1.mp4").mkdir()

            collected = HomeInterface._collect_supported_files(None, folder)

        self.assertEqual(
            [os.path.basename(path) for path in collected],
            ["episode2.mkv", "Episode10.MP4", "special.mp4"],
        )


if __name__ == "__main__":
    unittest.main()
