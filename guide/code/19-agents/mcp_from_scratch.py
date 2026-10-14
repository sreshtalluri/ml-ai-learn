# Adapted from AI Engineering from Scratch (MIT), phases/11-llm-engineering/14-model-context-protocol
"""A tiny MCP server and client in the standard library, talking JSON-RPC 2.0 over the stdio transport.

Follows the stateless request shape of MCP revision 2026-07-28: no initialize handshake; every request
carries its protocol version and client capabilities in params._meta. The server's tools return SYNTHETIC data.
This is a teaching subset (tools, resources, prompts, discovery, errors), not a conformant server.
Run from guide/:  uv run code/19-agents/mcp_from_scratch.py
(The script launches a copy of itself with --serve as the server subprocess.)
"""
import json
import subprocess
import sys
from pathlib import Path

VERSION = "2026-07-28"
META = "io.modelcontextprotocol/"

# ---------- server: three primitive types, registered as plain data + handlers ----------
TOOLS = {
    "get_order_status": {
        "description": "Look up one order by id. Returns status and carrier. Read-only.",
        "inputSchema": {"type": "object", "properties": {"order_id": {"type": "string"}},
                        "required": ["order_id"], "additionalProperties": False},
        "run": lambda order_id: {"A100": "shipped via UPS", "A101": "processing"}.get(order_id),
    },
}
RESOURCES = {"policy://returns": "Returns accepted within 30 days with receipt."}
PROMPTS = {"summarize_ticket": lambda ticket: f"Summarize this support ticket in two sentences:\n{ticket}"}


def error(rid, code, message, data=None):
    err = {"code": code, "message": message} | ({"data": data} if data else {})
    return {"jsonrpc": "2.0", "id": rid, "error": err}


def complete(payload, cacheable=False):
    out = {"resultType": "complete", **payload, "_meta": {META + "serverInfo": {"name": "shop", "version": "0.1.0"}}}
    return out | ({"ttlMs": 60_000, "cacheScope": "private"} if cacheable else {})


def handle(msg):
    """One JSON-RPC message in, one response (or None for a notification) out. No state survives between calls."""
    if "id" not in msg:
        return None                                              # notification: never answered
    rid, method, params = msg["id"], msg.get("method"), msg.get("params", {})
    meta = params.get("_meta", {})
    if not isinstance(meta.get(META + "protocolVersion"), str) or not isinstance(meta.get(META + "clientCapabilities"), dict):
        return error(rid, -32602, "missing protocolVersion or clientCapabilities in _meta")
    if meta[META + "protocolVersion"] != VERSION:
        return error(rid, -32022, "unsupported protocol version",
                     {"supported": [VERSION], "requested": meta[META + "protocolVersion"]})
    if method == "server/discover":
        result = complete({"supportedVersions": [VERSION],
                           "capabilities": {"tools": {}, "resources": {}, "prompts": {}}}, cacheable=True)
    elif method == "tools/list":
        tools = [{"name": n, "description": t["description"], "inputSchema": t["inputSchema"]} for n, t in sorted(TOOLS.items())]
        result = complete({"tools": tools}, cacheable=True)
    elif method == "tools/call":
        tool = TOOLS.get(params.get("name"))
        if tool is None:
            return error(rid, -32602, f"unknown tool {params.get('name')!r}")       # protocol error
        args = params.get("arguments", {})
        missing = [k for k in tool["inputSchema"]["required"] if k not in args]
        extra = [k for k in args if k not in tool["inputSchema"]["properties"]]
        if missing or extra:      # bad arguments are a TOOL error: the model reads it and can fix the call
            text, found = f"invalid arguments: missing {missing}, unexpected {extra}", None
        else:
            found = tool["run"](**args)
            text = json.dumps({"order_id": args["order_id"], "status": found}) if found else f"order {args['order_id']} not found; check the id"
        result = complete({"content": [{"type": "text", "text": text}], "isError": found is None})
    elif method == "resources/list":
        result = complete({"resources": [{"uri": u, "name": u.split("//")[1]} for u in RESOURCES]}, cacheable=True)
    elif method == "resources/read":
        uri = params.get("uri")
        if uri not in RESOURCES:
            return error(rid, -32602, f"resource not found: {uri}")
        result = complete({"contents": [{"uri": uri, "mimeType": "text/plain", "text": RESOURCES[uri]}]}, cacheable=True)
    elif method == "prompts/get":
        text = PROMPTS[params["name"]](**params.get("arguments", {}))
        result = complete({"messages": [{"role": "user", "content": {"type": "text", "text": text}}]})
    else:
        return error(rid, -32601, f"method not found: {method}")
    return {"jsonrpc": "2.0", "id": rid, "result": result}


def serve():
    """stdio transport: one JSON message per line on stdin, one per line on stdout. Logs go to stderr."""
    for line in sys.stdin:
        reply = handle(json.loads(line))
        if reply is not None:
            sys.stdout.write(json.dumps(reply) + "\n")
            sys.stdout.flush()


# ---------- client: launches the server as a subprocess and speaks to it over its stdin/stdout ----------
class Client:
    def __init__(self):
        self.proc = subprocess.Popen([sys.executable, __file__, "--serve"], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, text=True)
        self.next_id = 0

    def request(self, method, params=None, version=VERSION, show=False):
        self.next_id += 1
        p = dict(params or {})
        p["_meta"] = {META + "protocolVersion": version, META + "clientCapabilities": {},
                      META + "clientInfo": {"name": "course-client", "version": "1.0.0"}}
        msg = {"jsonrpc": "2.0", "id": self.next_id, "method": method, "params": p}
        line = json.dumps(msg)
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()
        reply = json.loads(self.proc.stdout.readline())
        assert reply["id"] == msg["id"]                       # responses are matched to requests by id
        if show:
            print(f"--> {line}\n<-- {json.dumps(reply)}")
        return reply, len(line)

    def close(self):
        self.proc.stdin.close()
        self.proc.wait()


def main():
    c = Client()
    print("=== discovery ===")
    disc, _ = c.request("server/discover", show=True)
    print("\n=== list tools, call one, read a resource, get a prompt ===")
    tools = c.request("tools/list")[0]["result"]["tools"]
    print("tools:", [t["name"] for t in tools])
    ok = c.request("tools/call", {"name": "get_order_status", "arguments": {"order_id": "A100"}})[0]["result"]
    print("call A100 ->", ok["content"][0]["text"], "| isError =", ok["isError"])
    miss = c.request("tools/call", {"name": "get_order_status", "arguments": {"order_id": "Z9"}})[0]["result"]
    print("call Z9   ->", miss["content"][0]["text"], "| isError =", miss["isError"])
    bad = c.request("tools/call", {"name": "get_order_status", "arguments": {"id": "A100"}})[0]["result"]
    print("bad args  ->", bad["content"][0]["text"], "| isError =", bad["isError"])
    res = c.request("resources/read", {"uri": "policy://returns"})[0]["result"]
    print("resource  ->", res["contents"][0]["text"])
    pr = c.request("prompts/get", {"name": "summarize_ticket", "arguments": {"ticket": "Box arrived crushed."}})[0]["result"]
    print("prompt    ->", pr["messages"][0]["content"]["text"].replace("\n", " | "))
    print("\n=== protocol errors (JSON-RPC error objects, not tool results) ===")
    for method, params, kw in [("tools/call", {"name": "delete_everything"}, {}),
                               ("tools/list", {}, {"version": "2025-11-25"}),
                               ("tools/destroy", {}, {})]:
        e = c.request(method, params, **kw)[0]["error"]
        print(f"{method:<11} -> {e['code']}: {e['message']}" + (f"  data={e['data']}" if "data" in e else ""))
    c.close()

    # The N x M integration arithmetic from the lesson.
    hosts, systems = 5, 8
    print(f"\nadapters without a protocol: {hosts} x {systems} = {hosts * systems}; "
          f"with MCP: {hosts} clients + {systems} servers = {hosts + systems}")
    schema_tokens = len(json.dumps(tools)) // 4  # ponytail: ~4 characters per token, a rough English/JSON proxy
    print(f"this tools/list is ~{schema_tokens} tokens; 3 servers x 12 tools x 150 tokens = {3 * 12 * 150} tokens re-sent every model call")
    figure(hosts, systems)
    assert disc["result"]["supportedVersions"] == [VERSION] and miss["isError"] and not ok["isError"]


def figure(hosts, systems):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from _style import BLUE, GRAY, INK, ORANGE, PURPLE, TEAL, plt, save
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [2.1, 1]})
    lanes = {"Model": 0.0, "Host + MCP client": 1.0, "MCP server": 2.0}
    for name, x in lanes.items():
        a1.plot([x, x], [0.4, 9.6], color=GRAY, lw=1, ls="--")
        a1.text(x, 9.9, name, ha="center", fontweight="bold", color=INK)
    steps = [  # (from, to, label, colour)
        (1, 2, "server/discover  (versions, capabilities)", BLUE),
        (1, 2, "tools/list  (names, descriptions, inputSchema)", BLUE),
        (1, 0, "tool schemas sent with the conversation", PURPLE),
        (0, 1, 'model emits tool call: get_order_status{"order_id":"A100"}', PURPLE),
        (1, 1, "host checks policy / asks user to approve", ORANGE),
        (1, 2, "tools/call  (name, arguments, _meta)", BLUE),
        (2, 1, "result: content[], isError=false", TEAL),
        (1, 0, "result appended to context; model continues", PURPLE),
    ]
    for i, (a, b, label, col) in enumerate(steps):
        y = 9 - i * 1.1
        if a == b:
            a1.annotate("", xy=(a + 0.18, y - 0.25), xytext=(a + 0.02, y + 0.1),
                        arrowprops=dict(arrowstyle="->", color=col, connectionstyle="arc3,rad=-1.2"))
            a1.text(a + 0.25, y - 0.05, label, fontsize=8.5, color=col, va="center")
            continue
        a1.annotate("", xy=(b, y), xytext=(a, y), arrowprops=dict(arrowstyle="->", color=col, lw=1.4))
        a1.text((a + b) / 2, y + 0.15, label, ha="center", fontsize=8.5, color=col,
                bbox=dict(facecolor="white", edgecolor="none", pad=0.5))
    a1.set(xlim=(-0.5, 2.6), ylim=(0.2, 10.4), title="One tool call through MCP")
    a1.axis("off")
    a2.bar(["custom glue\n(hosts x systems)", "MCP\n(hosts + systems)"], [hosts * systems, hosts + systems],
           color=[ORANGE, BLUE], width=0.55)
    for i, v in enumerate([hosts * systems, hosts + systems]):
        a2.text(i, v + 0.8, str(v), ha="center", fontweight="bold")
    a2.set(title=f"Adapters: {hosts} hosts, {systems} systems", ylabel="adapters")
    save(fig, "model-context-protocol")


if __name__ == "__main__":
    serve() if "--serve" in sys.argv else main()
