from enum import Enum
from pathlib import Path

from qfluentwidgets import getIconColor, Theme, FluentIconBase


class MyFluentIcon(FluentIconBase, Enum):
    Stop = "stop"

    def path(self, theme=Theme.AUTO):
        # getIconColor() return "white" or "black" according to current theme
        icon_dir = Path(__file__).resolve().parent
        return str(icon_dir / f"{self.value}_{getIconColor(theme)}.svg")
