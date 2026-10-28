"""MCP server exposing the fraud-investigation tools.

Run it standalone and point Claude Desktop (or any MCP client) at it:

    python -m p2_fraud.investigator.mcp_server

Claude Desktop config (~/Library/Application Support/Claude/claude_desktop_config.json):

    {
      "mcpServers": {
        "fraud-investigator": {
          "command": "/ABSOLUTE/PATH/ai-platforms/.venv/bin/python",
          "args": ["-m", "p2_fraud.investigator.mcp_server"],
          "cwd": "/ABSOLUTE/PATH/ai-platforms"
        }
      }
    }
"""
from __future__ import annotations

from . import tools

try:
    from fastmcp import FastMCP
except Exception:  # pragma: no cover
    from mcp.server.fastmcp import FastMCP  # fallback import path

mcp = FastMCP("fraud-investigator")

mcp.tool()(tools.get_transaction)
mcp.tool()(tools.get_user_history)
mcp.tool()(tools.score_transaction)
mcp.tool()(tools.explain_score)
mcp.tool()(tools.flag_transaction)


if __name__ == "__main__":
    mcp.run()        # stdio transport by default
