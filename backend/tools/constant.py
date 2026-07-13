from enum import Enum, unique

@unique
class InpaintMode(Enum):
    """
    图像重绘算法枚举
    """
    STTN_AUTO = "sttn-auto"
    STTN_DET = "sttn-det"
    LAMA = "lama"
    PROPAINTER = "propainter"
    OPENCV = "opencv"
    FIXED_WATERMARK = "fixed-watermark"
    MOVING_WATERMARK = "moving-watermark"
    SUBTITLE_FIXED_WATERMARK = "subtitle-fixed-watermark"
    SUBTITLE_MOVING_WATERMARK = "subtitle-moving-watermark"


SUBTITLE_INPAINT_MODES = frozenset((
    InpaintMode.STTN_AUTO,
    InpaintMode.STTN_DET,
    InpaintMode.LAMA,
    InpaintMode.PROPAINTER,
    InpaintMode.OPENCV,
    InpaintMode.SUBTITLE_FIXED_WATERMARK,
    InpaintMode.SUBTITLE_MOVING_WATERMARK,
))

FIXED_WATERMARK_INPAINT_MODES = frozenset((
    InpaintMode.FIXED_WATERMARK,
    InpaintMode.SUBTITLE_FIXED_WATERMARK,
))

MOVING_WATERMARK_INPAINT_MODES = frozenset((
    InpaintMode.MOVING_WATERMARK,
    InpaintMode.SUBTITLE_MOVING_WATERMARK,
))


def uses_subtitles(mode):
    """Return whether an inpaint mode includes subtitle removal."""
    return mode in SUBTITLE_INPAINT_MODES


def uses_fixed_watermark(mode):
    """Return whether an inpaint mode includes a fixed watermark."""
    return mode in FIXED_WATERMARK_INPAINT_MODES


def uses_moving_watermark(mode):
    """Return whether an inpaint mode includes a tracked moving watermark."""
    return mode in MOVING_WATERMARK_INPAINT_MODES


@unique
class SubtitleDetectMode(Enum):
    """
    字幕检测算法枚举
    """
    PP_OCRv5_MOBILE = "PP_OCRv5_MOBILE"
    PP_OCRv5_SERVER = "PP_OCRv5_SERVER"
