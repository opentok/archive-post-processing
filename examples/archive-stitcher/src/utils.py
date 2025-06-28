# copyright 2025 Vonage

import subprocess
import sys
import tempfile

from pathlib import Path


FFMPEG: str = 'ffmpeg'
FFPROBE: str = 'ffprobe'


class StitcherException(Exception):
    pass


def printerr(*args, **kwargs):
    print(*args, **kwargs, file=sys.stderr)


def raise_error(msg: str):
    printerr(msg)
    raise StitcherException(msg)


def run_exec(*args, get_stderr=False) -> str:
    try:
        printerr('running command: ', *args)

        result = subprocess.run(
            map(str, args),
            text=True,
            capture_output=True,
            check=True
        )
        return result.stdout if not get_stderr else result.stderr
    except subprocess.CalledProcessError as e:
        raise_error(f'Command "{args}" failed with status {e.returncode}\n\nstdout: {e.stdout}\n\nstderr: {e.stderr}')


def create_tempfile(suffix: str, dir: Path) -> Path:
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=dir) as tmp:
        return Path(tmp.name)
