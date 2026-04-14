from __future__ import annotations

import asyncio
import atexit


def install_s3_cleanup_handler() -> None:
    """Work around s3fs/aiobotocore double-close at interpreter shutdown.

    aiobotocore 3.x (PR #1402) added a non-idempotent assert in
    AIOHTTPSession.__aexit__. s3fs registers a weakref finalizer that
    can trigger __aexit__ a second time at process exit, raising
    "Session was never entered". This installs two mitigations:

    1. An atexit handler that eagerly clears the fsspec instance cache.
    2. An asyncio exception handler that silences only this specific error.

    References:
        https://github.com/fsspec/s3fs/issues/1001
        https://github.com/aio-libs/aiobotocore/pull/1402
    """

    def _suppress_session_cleanup(
        loop: asyncio.AbstractEventLoop,
        context: dict[str, object],
    ) -> None:
        exception = context.get("exception")
        if isinstance(exception, AssertionError) and "Session was never entered" in str(exception):
            return
        loop.default_exception_handler(context)

    def _clear_fsspec_cache() -> None:
        try:
            from fsspec import AbstractFileSystem

            AbstractFileSystem.clear_instance_cache()
        except Exception:
            pass

    try:
        loop = asyncio.get_event_loop()
        loop.set_exception_handler(_suppress_session_cleanup)
    except RuntimeError:
        pass
    atexit.register(_clear_fsspec_cache)
