"""Client for the vendored medical-mcp server (FDA, WHO, PubMed, RxNorm, AAP).

The MCP Python SDK is async and the server is a Node process spoken to over stdio,
while every graph node here is sync. This module owns that bridge so the node itself
stays synchronous like the rest of the graph.

The server is spawned per call. That costs a process launch (~0.5s handshake) each
turn, but a long-lived session would have to survive across the graph's worker threads
and outlive a single request — not worth the complexity until the latency matters.
"""
import asyncio

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from app.config import MCP_SERVER_DIR, MCP_TIMEOUT

# Operational tooling, not clinical evidence — never offer it as a retrieval option.
EXCLUDED_TOOLS = {"get-cache-stats"}


def _server_params() -> StdioServerParameters:
    return StdioServerParameters(
        command="node",
        args=[f"{MCP_SERVER_DIR}/build/index.js"],
        cwd=MCP_SERVER_DIR,
    )


async def _list_tools_async() -> list[dict]:
    async with stdio_client(_server_params()) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = (await session.list_tools()).tools

    return [
        {
            "name": t.name,
            "description": t.description or "",
            "schema": t.inputSchema or {},
        }
        for t in tools
        if t.name not in EXCLUDED_TOOLS
    ]


async def _call_tool_async(tool_name: str, arguments: dict) -> str:
    async with stdio_client(_server_params()) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, arguments)

    if not result.content:
        return ""

    # Tool results are a list of content blocks; only the text ones carry evidence.
    return "\n".join(block.text for block in result.content if hasattr(block, "text"))


def _run(coro):
    """Run an async MCP call from a sync node, under the session timeout.

    Graph nodes are invoked from threads with no running event loop (FastAPI runs the
    sync route in its threadpool, and LangGraph dispatches sync nodes to workers), so
    a fresh loop per call is safe.
    """
    return asyncio.run(asyncio.wait_for(coro, timeout=MCP_TIMEOUT))


def list_tools() -> list[dict]:
    """Available evidence tools, with their names, descriptions, and arg schemas."""
    return _run(_list_tools_async())


def call_tool(tool_name: str, arguments: dict) -> str:
    """Call one tool and return its text output.

    Raises on an unknown tool name (the server responds with an MCP error rather than
    an error result), on timeout, and if the server process cannot be started.
    """
    return _run(_call_tool_async(tool_name, arguments))
