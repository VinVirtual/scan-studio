"""Errors the command line turns into exit codes and one plain sentence."""
from __future__ import annotations


class ScanError(Exception):
    """A problem with the input or the capture that the user can fix. Exit code 1."""
    exit_code = 1


class UsageError(ScanError):
    """A wrong option or argument. Exit code 2."""
    exit_code = 2


class BudgetError(ScanError):
    """The input is bigger than the limits allow (pixels, frames, bytes). Exit code 3."""
    exit_code = 3


class FacesFound(ScanError):
    """People were found in the frames. scan-studio is for objects only. Exit code 4."""
    exit_code = 4
