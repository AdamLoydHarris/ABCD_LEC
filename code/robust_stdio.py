"""Make stdout/stderr writes wait on a stalled filesystem instead of raising.

Written 2026-10-02. ly07's 6-day sort (3666095) died 2026-10-01 00:00 UTC when tqdm's progress
bar, inside kilosort's template_matching.extract, wrote to stderr: the fd was non-blocking, a
ceph write stalled, and the write raised `BlockingIOError: [Errno 11]` (EAGAIN) instead of
waiting. Kilosort died on it, then the error path hung writing the traceback, and SLURM showed
the job RUNNING for 33 h on an idle A100. The other three concurrent sorts had 0 such errors, so
it was a transient stall that hit one job.

Import first, before kilosort/tqdm/logging set up their streams:

    import robust_stdio  # noqa: F401

I/O: forces fds 1 and 2 blocking, and replaces sys.stdout/sys.stderr with wrappers that retry a
write on EAGAIN for up to 5 min, then drop that one log write rather than kill the job.
"""
from __future__ import annotations

import os
import sys
import time

_RETRY_S = 0.5
_MAX_TRIES = 600          # 5 min of retrying one write before giving up on the log line


class _RetryOnEAGAIN:
    """File-like proxy: retry write/flush on BlockingIOError, delegate everything else."""

    def __init__(self, stream):
        self._stream = stream

    def write(self, data):
        for _ in range(_MAX_TRIES):
            try:
                return self._stream.write(data)
            except BlockingIOError:
                time.sleep(_RETRY_S)
        return 0  # losing a progress-bar line is fine; losing the sort is not

    def flush(self):
        for _ in range(_MAX_TRIES):
            try:
                return self._stream.flush()
            except BlockingIOError:
                time.sleep(_RETRY_S)

    def __getattr__(self, name):
        return getattr(self._stream, name)


def apply() -> None:
    for fd in (1, 2):
        try:
            os.set_blocking(fd, True)
        except OSError:
            pass
    if not isinstance(sys.stdout, _RetryOnEAGAIN):
        sys.stdout = _RetryOnEAGAIN(sys.stdout)
    if not isinstance(sys.stderr, _RetryOnEAGAIN):
        sys.stderr = _RetryOnEAGAIN(sys.stderr)


apply()
