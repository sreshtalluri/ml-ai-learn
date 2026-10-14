<!-- GENERATED from model-context-protocol.yml by web/scripts/gen-quizzes.mjs. Edit the .yml, then run: npm run gen:quizzes -->
# Quiz: Model Context Protocol

Covers the lesson [Model Context Protocol](../lessons/19-agents/04-model-context-protocol.md). Try each question before opening the answer. The [website](https://sreshtalluri.github.io/ml-ai-learn/quizzes/model-context-protocol/) grades these interactively and tracks a review queue.

## 1. Multiple choice (easy)

In an MCP setup, which component sends `tools/call` to the server?

- **A.** The language model, directly
- **B.** The MCP client inside the host application, after the model emits a tool call
- **C.** The server, to itself
- **D.** The user's browser

<details>
<summary>Answer</summary>

**B.** The MCP client inside the host application, after the model emits a tool call

The model only emits a tool call in its output. The host decides whether to allow it and its MCP client sends the JSON-RPC request.

- **A:** The model never talks to the server; it only produces text the host parses.
- **B:** Correct.
- **C:** Servers answer requests; clients send them.
- **D:** The browser is not part of the protocol path.

</details>

## 2. Match (easy)

Match each server primitive to who typically decides to use it.

| Concept | Options |
|---|---|
| Tools | The user |
| Resources | The application |
| Prompts | The model (with host approval) |

<details>
<summary>Answer</summary>

- Tools → The model (with host approval)
- Resources → The application
- Prompts → The user

Tools are model-controlled actions, resources are application-controlled data addressed by URI, and prompts are user-chosen templates.

</details>

## 3. Multiple choice (easy)

What message format does MCP use?

- **A.** gRPC with protobuf
- **B.** JSON-RPC 2.0 requests, responses, and notifications
- **C.** GraphQL queries
- **D.** Plain REST with one URL per tool

<details>
<summary>Answer</summary>

**B.** JSON-RPC 2.0 requests, responses, and notifications

Every MCP message is JSON-RPC 2.0, carried over stdio or Streamable HTTP.

- **A:** MCP does not use gRPC.
- **B:** Correct.
- **C:** MCP does not use GraphQL.
- **D:** Streamable HTTP uses one endpoint for all methods, with the method in the JSON body.

</details>

## 4. Multiple choice (medium)

A local stdio MCP server works in tests but the host reports parse errors after you add debug logging. What is the likely cause?

- **A.** The server logs to stdout, which is reserved for protocol messages
- **B.** The server logs to stderr
- **C.** JSON-RPC forbids logging
- **D.** The server is missing an `initialize` handler

<details>
<summary>Answer</summary>

**A.** The server logs to stdout, which is reserved for protocol messages

Under stdio every line on stdout must be a valid MCP message. Logs belong on stderr.

- **A:** Correct.
- **B:** Stderr is the right place for logs and does not corrupt the stream.
- **C:** Logging is fine; only where it goes matters.
- **D:** That would produce an error response, not unparseable output, and 2026-07-28 servers don't use initialize.

</details>

## 5. Multiple choice (medium)

A tool is called with a valid tool name but an order id that does not exist. How should the server report it?

- **A.** A JSON-RPC error with code -32601
- **B.** A JSON-RPC error with code -32602
- **C.** A normal result with `isError` true and a message the model can act on
- **D.** Close the connection

<details>
<summary>Answer</summary>

**C.** A normal result with `isError` true and a message the model can act on

The request was well-formed and the tool ran; the failure is something the model can fix, so it goes in a result the model reads.

- **A:** -32601 means the method itself does not exist.
- **B:** Protocol errors are for broken requests such as an unknown tool; the host may not show them to the model.
- **C:** Correct.
- **D:** Nothing is wrong with the transport.

</details>

## 6. Multiple choice (medium)

In MCP revision 2026-07-28, where does a client put its protocol version and capabilities?

- **A.** In an `initialize` request sent once per connection
- **B.** In `params._meta` on every request
- **C.** In a cookie set by the server
- **D.** Nowhere; versions are negotiated by the transport

<details>
<summary>Answer</summary>

**B.** In `params._meta` on every request

The 2026-07-28 revision removed the initialize handshake. Every request carries `io.modelcontextprotocol/protocolVersion` and client capabilities in `_meta`, so any server instance can answer it.

- **A:** That was how revisions up to 2025-11-25 worked.
- **B:** Correct.
- **C:** MCP does not use cookies for protocol state.
- **D:** Each request declares its version; on HTTP it is also mirrored in a header.

</details>

## 7. Multiple choice (medium)

Your MCP server manages shopping carts and runs on several instances behind a load balancer. How should it keep each cart across calls?

- **A.** Keep the cart in memory keyed by the TCP connection
- **B.** Return an opaque cart id from a create tool, accept it as an argument later, store carts in a shared store, and check authorization on every call
- **C.** Ask the model to resend the whole cart contents on every call
- **D.** Use the server's name from serverInfo as the key

<details>
<summary>Answer</summary>

**B.** Return an opaque cart id from a create tool, accept it as an argument later, store carts in a shared store, and check authorization on every call

The protocol has no session, so state lives behind an explicit handle passed as an ordinary tool argument.

- **A:** Requests can land on any instance, and the protocol does not tie state to connections.
- **B:** Correct.
- **C:** That is fragile, wastes tokens, and lets the model alter the cart.
- **D:** serverInfo is self-reported identity of the server, not of a user's cart.

</details>

## 8. Calculation (easy)

Six host applications each need to reach nine systems. How many fewer integrations do you build with MCP (hosts + systems) than with custom adapters (hosts × systems)?

<details>
<summary>Answer</summary>

**39**

Custom $= 6 \times 9 = 54$. MCP $= 6 + 9 = 15$. Difference $= 54 - 15 = 39$.

</details>

## 9. Calculation (medium)

An agent connects 4 servers with 8 tools each; each tool definition is about 125 tokens. How many input tokens go to tool definitions over a 10-call run, before caching?

<details>
<summary>Answer</summary>

**40000**

Per call $4 \times 8 \times 125 = 4{,}000$ tokens. Over 10 calls $10 \times 4{,}000 = 40{,}000$.

</details>

## 10. Multiple choice (hard)

After installing a third-party MCP server, your coding agent starts including the contents of `~/.ssh/config` in summaries. No tool was called with that path by the user. What is the most likely mechanism?

- **A.** The model was fine-tuned on SSH configs
- **B.** Instructions embedded in the server's tool descriptions or results, which land in the model's context as untrusted text
- **C.** A bug in JSON-RPC id matching
- **D.** Prefix caching leaked another user's prompt

<details>
<summary>Answer</summary>

**B.** Instructions embedded in the server's tool descriptions or results, which land in the model's context as untrusted text

Tool descriptions and outputs are text from the server. A malicious or compromised server can inject instructions; the host must treat them as untrusted and gate sensitive reads.

- **A:** Training data does not explain behaviour that started after installing a server.
- **B:** Correct.
- **C:** Mismatched ids cause wrong pairing of replies, not new file reads.
- **D:** The file comes from this machine, read by the agent's own tools.

</details>
