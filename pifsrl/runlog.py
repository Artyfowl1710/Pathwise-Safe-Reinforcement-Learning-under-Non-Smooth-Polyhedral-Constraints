"""
Run logging: keep the console clean, keep EVERYTHING in a log file.

After `start()`:
  * every normal print, every Python warning/traceback, AND the C-level
    messages written by native libraries (e.g. the Intel oneMKL "SLASWP"
    lines) go to  logs/<name>_<timestamp>.log   -- file descriptors 1 and 2 are
    redirected at the OS level, so native output is captured too.
  * only messages sent through `say()` appear on the screen (and they are also
    copied into the log, so the log is a complete record).

If `start()` is never called (e.g. under pytest, or with --verbose), `say()` is
just `print()`, so library code can call it unconditionally.

This module imports only the standard library so it can be loaded BEFORE
numpy/torch (whose import-time warnings are then captured as well).
"""
from __future__ import annotations

import faulthandler
import os
import sys
import threading
import time

_console = None          # the original console, as a text stream
_logf = None             # the log file
_log_path = None
_timer = None


def start(name: str, log_dir: str = "logs") -> str:
    """Redirect stdout/stderr (Python AND native) to a new log file."""
    global _console, _logf, _log_path
    os.makedirs(log_dir, exist_ok=True)
    _log_path = os.path.join(log_dir, f"{name}_{time.strftime('%Y%m%d_%H%M%S')}.log")
    sys.stdout.flush()
    sys.stderr.flush()
    enc = getattr(sys.stdout, "encoding", None) or "utf-8"
    _console = os.fdopen(os.dup(1), "w", encoding=enc, errors="replace", buffering=1)
    _logf = open(_log_path, "w", encoding="utf-8", errors="replace", buffering=1)
    os.dup2(_logf.fileno(), 1)
    os.dup2(_logf.fileno(), 2)
    sys.stdout = _logf
    sys.stderr = _logf
    return _log_path


def log_path() -> str | None:
    return _log_path


def say(msg: str = "", *_ignored, **_kw) -> None:   # accepts print-style kwargs (flush=...)
    """An IMPORTANT message: shown on screen and recorded in the log."""
    print(msg)                       # -> log file (or console if not started)
    if _console is not None:
        try:
            _console.write(msg + "\n")
            _console.flush()
        except Exception:
            pass


def watchdog(seconds: float, what: str = "run") -> None:
    """If the run is still going after `seconds`, dump every thread's stack to
    the log (repeating) and tell the user once. Diagnoses a hang without any
    action from the user."""
    global _timer
    if _logf is None:
        return
    faulthandler.dump_traceback_later(seconds, repeat=True, file=_logf)

    def _notice():
        say(f"[note] {what} is taking longer than {seconds:.0f}s. A stack dump is being "
            f"written to the log: {_log_path}\n"
            f"       If it keeps going, press Ctrl+C and send that log file.")
    _timer = threading.Timer(seconds, _notice)
    _timer.daemon = True
    _timer.start()


def stop_watchdog() -> None:
    try:
        faulthandler.cancel_dump_traceback_later()
    except Exception:
        pass
    if _timer is not None:
        _timer.cancel()


def finish(code: int = 0) -> None:
    """Flush everything and exit WITHOUT interpreter teardown.

    Defensive measure: native thread pools (OpenMP/MKL) can hang during normal
    interpreter shutdown on some Windows setups. All results are written before
    this is called, and the log/console are flushed here.
    """
    stop_watchdog()
    for s in (_logf, _console, sys.__stdout__, sys.__stderr__):
        try:
            if s is not None:
                s.flush()
        except Exception:
            pass
    os._exit(code)
