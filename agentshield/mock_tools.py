"""
Mock tool registry for offline testing and demo verification.
Simulates tool outputs without side-effects.

For real-world integrations the actual tool registry comes from the agent framework
(LangChain, AutoGen, CrewAI). The helpers here are used by tests, the demo
runner, and the policy engine's category resolver.
"""

from __future__ import annotations

import re
from typing import Any, Callable

# ── Exact-match category map (high-confidence, fast path) ──────────────────
TOOL_CATEGORIES: dict[str, str] = {
    "read_file": "filesystem",
    "write_file": "filesystem",
    "delete_file": "filesystem",
    "list_files": "filesystem",
    "compress": "filesystem",
    "http_request": "network",
    "get_status": "network",
    "web_search": "network",
    "fetch_url": "network",
    "curl": "network",
    "send_email": "email",
    "send_message": "email",
    "execute_code": "code",
    "run_python": "code",
    "run_shell": "code",
    "bash": "code",
    "execute_tool": "system",
    "create_calendar_event": "calendar",
    "update_user_profile": "system",
    "list_orders": "database",
    "get_order": "database",
    "sql_query": "database",
    "db_query": "database",
}

# ── Fuzzy-match rules (keyword substrings → category) ──────────────────────
# Applied left-to-right; first match wins.
_FUZZY_RULES: list[tuple[str, str]] = [
    # filesystem
    ("read_file", "filesystem"), ("write_file", "filesystem"),
    ("delete_file", "filesystem"), ("list_file", "filesystem"),
    ("file_read", "filesystem"), ("file_write", "filesystem"),
    ("file_delete", "filesystem"), ("file_list", "filesystem"),
    ("open_file", "filesystem"), ("save_file", "filesystem"),
    ("compress", "filesystem"), ("extract", "filesystem"),
    ("mkdir", "filesystem"), ("rmdir", "filesystem"),
    ("move_file", "filesystem"), ("copy_file", "filesystem"),
    # network
    ("http", "network"), ("https", "network"), ("fetch", "network"),
    ("request", "network"), ("get_url", "network"), ("post_url", "network"),
    ("curl", "network"), ("wget", "network"), ("web_search", "network"),
    ("download", "network"), ("upload", "network"), ("socket", "network"),
    # email
    ("email", "email"), ("mail", "email"), ("smtp", "email"),
    ("send_message", "email"), ("slack_message", "email"),
    ("teams_message", "email"), ("notify", "email"),
    # code / execution
    ("exec", "code"), ("run_", "code"), ("_run", "code"),
    ("bash", "code"), ("shell", "code"), ("python", "code"),
    ("compile", "code"), ("eval", "code"), ("script", "code"),
    # database
    ("sql", "database"), ("db_", "database"), ("_db", "database"),
    ("query", "database"), ("insert", "database"), ("update_record", "database"),
    ("delete_record", "database"), ("mongo", "database"), ("redis", "database"),
    # system / agent management
    ("mcp", "system"), ("register_tool", "system"), ("tool_use", "system"),
    ("agent_call", "system"), ("spawn_", "system"), ("sub_agent", "system"),
    ("spawn_agent", "system"), ("tool_registration", "system"),
    ("update_profile", "system"), ("set_config", "system"),
    # calendar
    ("calendar", "calendar"), ("schedule", "calendar"),
    ("create_event", "calendar"), ("update_event", "calendar"),
]


def get_tool_category(tool_name: str) -> str:
    """Resolve tool category with fuzzy matching for real-world tool names.

    Resolution order:
    1. Exact match in TOOL_CATEGORIES
    2. Fuzzy substring match via _FUZZY_RULES
    3. Falls back to "unknown"
    """
    if not tool_name:
        return "unknown"

    # 1. Exact match (fast path)
    exact = TOOL_CATEGORIES.get(tool_name)
    if exact:
        return exact

    # 2. Fuzzy keyword match (case-insensitive)
    lower = tool_name.lower()
    for keyword, category in _FUZZY_RULES:
        if keyword in lower:
            return category

    return "unknown"


# Mock implementation functions
def mock_read_file(path: str) -> str:
    if "passwd" in path or "shadow" in path:
        return "root:x:0:0:root:/root:/bin/bash\ndaemon:x:1:1:daemon:/usr/sbin:/usr/sbin/nologin"
    if "id_rsa" in path:
        return "-----BEGIN OPENSSH PRIVATE KEY-----\nb3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAAEbm9uZQAAAAAAAAABAAAAMwAAAAtzc2gtZW\n2Q1NTE5AAAAIC43Y3pGSHBvaXpKTnJxeFpoL3FpL0ZzRGVzVjNaWk\n-----END OPENSSH PRIVATE KEY-----"
    if "report" in path or "financial" in path:
        return "[MOCK] CONFIDENTIAL Q3 Financial Report: Revenue = $4.2M, Net Income = $1.1M. Operating costs reduced."
    return f"[MOCK] Read file content of {path} successfully."


def mock_write_file(path: str, content: str = "") -> str:
    return f"[MOCK] Wrote {len(content)} characters to file: {path}."


def mock_delete_file(path: str) -> str:
    return f"[MOCK] Target file at {path} would be permanently deleted."


def mock_list_files(directory: str) -> str:
    return f"[MOCK] Directory list of {directory}: ['q3_report.pdf', 'briefing.txt', 'main.py', 'utils.py']"


def mock_compress(files: list[str], output: str) -> str:
    return f"[MOCK] Compressed {len(files)} files into archive: {output}."


def mock_http_request(url: str, method: str = "GET", data: str = "") -> str:
    return f"[MOCK] HTTP 200 OK Response from {url} (Payload len: {len(data)})"


def mock_get_status(endpoint: str) -> str:
    return f"[MOCK] API Endpoint '{endpoint}' Status: 200 Healthy, Response Time: 42ms"


def mock_send_email(to: str, body: str) -> str:
    return f"[MOCK] Outbound email successfully dispatched to {to}."


def mock_execute_code(code: str) -> str:
    return f"[MOCK] Python interpreter executed successfully. Stdout: 'Tests passed.'"


def mock_execute_tool(server: str, action: str, params: dict[str, Any] = None) -> str:
    return f"[MOCK] Executed action '{action}' on MCP server '{server}' with params: {params or {}}"


def mock_create_calendar_event(title: str, attendees: list[str] = None) -> str:
    return f"[MOCK] Calendar event '{title}' created with attendees: {attendees or []}."


def mock_update_user_profile(display_name: str = None, bio: str = None) -> str:
    return f"[MOCK] User profile updated (display_name: {display_name}, bio: {bio})"


def mock_list_orders() -> str:
    return "[MOCK] Database returned 3 orders: [#1001 (Pending), #1002 (Shipped), #1003 (Cancelled)]"


def mock_get_order(id: int) -> str:
    return f"[MOCK] Database return for order #{id}: Item: 'Enterprise AI Agent Framework License', Status: 'Active'"


# Master Registry
MOCK_TOOLS: dict[str, Callable[..., Any]] = {
    "read_file": mock_read_file,
    "write_file": mock_write_file,
    "delete_file": mock_delete_file,
    "list_files": mock_list_files,
    "compress": mock_compress,
    "http_request": mock_http_request,
    "get_status": mock_get_status,
    "send_email": mock_send_email,
    "execute_code": mock_execute_code,
    "execute_tool": mock_execute_tool,
    "create_calendar_event": mock_create_calendar_event,
    "update_user_profile": mock_update_user_profile,
    "list_orders": mock_list_orders,
    "get_order": mock_get_order,
}


def execute_mock_tool(tool_name: str, **kwargs: Any) -> str:
    """Lookup and execute a mock tool function, returning a response string."""
    func = MOCK_TOOLS.get(tool_name)
    if not func:
        return f"[MOCK] Tool '{tool_name}' executed with arguments: {kwargs}"

    # Handle argument mapping
    try:
        import inspect
        sig = inspect.signature(func)
        bound = sig.bind_partial(**kwargs)
        return str(func(*bound.args, **bound.kwargs))
    except Exception as e:
        return f"[MOCK ERROR] Tool '{tool_name}' execution error: {str(e)}"
