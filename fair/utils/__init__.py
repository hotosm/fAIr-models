from __future__ import annotations

import asyncio
import atexit


def install_s3_cleanup_handler() -> None:

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
