import os
import re
import unicodedata


_CHINESE_DIGITS = {
    "零": 0,
    "〇": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}
_CHINESE_SMALL_UNITS = {"十": 10, "百": 100, "千": 1000}
_CHINESE_LARGE_UNITS = {"万": 10_000, "亿": 100_000_000}
_CHINESE_EPISODE_PATTERN = re.compile(
    r"第\s*([零〇一二两三四五六七八九十百千万亿]+)\s*([集话話回期章])"
)
_NATURAL_NUMBER_PATTERN = re.compile(r"(\d+)")


def chinese_number_to_int(text):
    """Convert a common Chinese episode numeral to an integer."""
    if not text:
        return None

    has_unit = any(
        char in _CHINESE_SMALL_UNITS or char in _CHINESE_LARGE_UNITS
        for char in text
    )
    if not has_unit:
        try:
            return int("".join(str(_CHINESE_DIGITS[char]) for char in text))
        except (KeyError, ValueError):
            return None

    total = 0
    section = 0
    number = 0
    for char in text:
        if char in _CHINESE_DIGITS:
            number = _CHINESE_DIGITS[char]
            continue
        if char in _CHINESE_SMALL_UNITS:
            section += (number or 1) * _CHINESE_SMALL_UNITS[char]
            number = 0
            continue
        if char in _CHINESE_LARGE_UNITS:
            section = (section + number) * _CHINESE_LARGE_UNITS[char]
            total += section
            section = 0
            number = 0
            continue
        return None
    return total + section + number


def episode_name_sort_key(path):
    """Build a deterministic natural-sort key for episode file names."""
    basename = os.path.basename(os.fspath(path))
    stem, extension = os.path.splitext(basename)
    normalized = unicodedata.normalize("NFKC", stem).casefold()
    normalized_extension = unicodedata.normalize("NFKC", extension).casefold()

    def replace_chinese_episode(match):
        episode_number = chinese_number_to_int(match.group(1))
        if episode_number is None:
            return match.group(0)
        return f"第{episode_number}{match.group(2)}"

    normalized = _CHINESE_EPISODE_PATTERN.sub(replace_chinese_episode, normalized)
    natural_parts = tuple(
        (0, int(part), len(part)) if part.isdigit() else (1, part, 0)
        for part in _NATURAL_NUMBER_PATTERN.split(normalized)
        if part
    )
    return natural_parts, normalized, normalized_extension, basename
