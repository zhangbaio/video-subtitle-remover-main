import argparse
from enum import Enum

from .constant import InpaintMode

def parse_args():
    parser = argparse.ArgumentParser(
        description="Video Subtitle Remover Command Line Tool"
    )
    parser.add_argument(
        "--input", "-i", required=True, type=str,
        help="Input video file path"
    )
    parser.add_argument(
        "--output", "-o", required=False, type=str, default=None,
        help="Output video file path (optional)"
    )
    parser.add_argument(
        "--subtitle-area-coords", "-c", action="append", nargs=4, type=int, metavar=("YMIN", "YMAX", "XMIN", "XMAX"),
        help="Subtitle area coordinates (ymin ymax xmin xmax). Can be specified multiple times for multiple areas."
    )
    parser.add_argument(
        "--watermark-area-coords", "-w", action="append", nargs=4, type=int,
        metavar=("YMIN", "YMAX", "XMIN", "XMAX"),
        help="Watermark area coordinates (ymin ymax xmin xmax). Can be specified multiple times for fixed watermarks."
    )
    parser.add_argument(
        "--watermark-reference-frame", type=int, default=0,
        help="Zero-based reference frame used by moving-watermark modes (default: 0)."
    )
    parser.add_argument(
        "--watermark-template-source", type=str, default=None,
        help="Optional video containing the moving-watermark reference template."
    )
    parser.add_argument(
        "--inpaint-mode", type=str, default="sttn-auto",
        choices=[mode.name.lower().replace('_','-') for mode in InpaintMode],
        help="Inpaint mode, default is sttn-auto"
    )
    args = parser.parse_args()
    args.inpaint_mode = InpaintMode[args.inpaint_mode.replace('-','_').upper()]
    if args.subtitle_area_coords is None:
        args.subtitle_area_coords = []
    if args.watermark_area_coords is None:
        args.watermark_area_coords = []
    return args
