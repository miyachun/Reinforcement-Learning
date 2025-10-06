#npx @modelcontextprotocol/inspector python a.py
from fastmcp import FastMCP

mcp = FastMCP("my-server")

@mcp.tool()
def hello(name: str) -> str:
    """打招呼"""
    return f"你好，{name}！"

if __name__ == "__main__":
    mcp.run()