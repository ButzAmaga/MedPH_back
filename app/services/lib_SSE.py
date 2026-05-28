# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------
import asyncio
import json
from typing import Any

from services.custom_toJson import SafeJsonEncoder

# Interval (seconds) between keep-alive heartbeats during long blocking operations
HEARTBEAT_INTERVAL = 10

def sse(event: str, data: dict) -> str:
    """Formats a single Server-Sent Events frame, safely serialising all cudf/cupy types."""
    return f"event: {event}\ndata: {json.dumps(data, cls=SafeJsonEncoder)}\n\n"
 

def sse_progress(message: str, step: int, total_steps: int = 6) -> str:
    return sse("progress", {"step": step, "total": total_steps, "message": message})


def sse_heartbeat() -> str:
    """SSE comment line — invisible to EventSource listeners but resets proxy timeout."""
    return ": heartbeat\n\n"


# ---------------------------------------------------------------------------
# Concurrent heartbeat runner
# ---------------------------------------------------------------------------

async def run_with_heartbeats(queue: asyncio.Queue, fn, *args) -> Any:
    """
    Runs a blocking function in a thread pool executor.
    While it is running, pushes a heartbeat into `queue` every
    HEARTBEAT_INTERVAL seconds so the stream generator can forward
    them to the client without blocking on the heavy work.
    """
    loop = asyncio.get_event_loop()
    task = loop.run_in_executor(None, fn, *args)

    while not task.done():
        try:
            # Wait up to HEARTBEAT_INTERVAL seconds; if task finishes sooner,
            # asyncio.wait raises TimeoutError which we just swallow.
            await asyncio.wait_for(asyncio.shield(task), timeout=HEARTBEAT_INTERVAL)
        except asyncio.TimeoutError:
            # Task still running — push a heartbeat for the generator to yield
            await queue.put(sse_heartbeat())

    return await task  # re-raises any exception from the thread