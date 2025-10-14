# copyright 2025 Vonage

import subprocess
import sys
import tempfile

from pathlib import Path


FFMPEG: str = 'ffmpeg'
FFPROBE: str = 'ffprobe'


class StitcherException(Exception):
    """
    Custom exception for Archive Stitcher errors.
    """
    pass


def printerr(*args, **kwargs) -> None:
    """
    Print messages to stderr.

    Args:
        *args: Arguments to print.
        **kwargs: Keyword arguments to print.
    """
    print(*args, **kwargs, file=sys.stderr)


def raise_error(msg: str) -> None:
    """
    Print error message and raise a StitcherException.

    Args:
        msg (str): Error message.

    Raises:
        StitcherException: Always raised with the provided message.
    """
    printerr(msg)
    raise StitcherException(msg)


def run_exec(*args, get_stderr=False) -> str:
    """
    Run a subprocess command and return its output.

    Args:
        *args: Command arguments.
        get_stderr (bool): If True, return stderr instead of stdout.

    Returns:
        str: Command output (stdout or stderr).

    Raises:
        StitcherException: If the command fails.
    """
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
    """
    Create a temporary file and return its path.

    Args:
        suffix (str): Suffix to use for the file.
        dir (Path): Directory in which to create the file.

    Returns:
        Path: Path to the created temporary file.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=dir) as tmp:
        return Path(tmp.name)
