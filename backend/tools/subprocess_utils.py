import os
import subprocess


def hidden_subprocess_kwargs():
    """Return subprocess options that keep GUI-owned commands invisible.

    Console executables launched from ``pythonw.exe`` can otherwise request a
    new console.  When Windows Terminal is the default console host, that
    appears as a large black window for every FFmpeg invocation.
    """
    if os.name != "nt":
        return {}

    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startupinfo.wShowWindow = subprocess.SW_HIDE
    return {
        "creationflags": subprocess.CREATE_NO_WINDOW,
        "startupinfo": startupinfo,
    }
