# Toolify Detailed Design

## 1. Overview

**Toolify** is the **centralized functional tool manager** for the entire multi-agent ecosystem.
It provides:

* **Tool Registration** — Static (config-based) and Dynamic (runtime API).
* **Tool Orchestration** — Mapping tasks to the correct tools based on context.
* **Tool Execution Layer** — Abstracted interface to invoke tools across different formats (LangChain tools, MCP tools, REST APIs, local scripts, etc.).
* **Dependency Graph Management** — Maintains an execution DAG for tools to resolve order of operations.
* **Integration with Orchestrix** — Supplies execution-ready tool plans to Orchestrix for goal fulfillment.

Toolify is **tool-agnostic**, meaning it doesn't execute the *high-level* reasoning itself but manages tool metadata, connectivity, and execution rules.

---

## 2. Component Architecture

### 2.1 Registry Layer

Manages metadata and configurations for each tool.

#### Responsibilities:

* Maintain **tool registry** (unique ID, name, type, capabilities, version).
* Support **static registration** (via config files or bootstrapping scripts).
* Support **dynamic registration** (via API calls during runtime).
* Store tool **capabilities** in structured form (input schema, output schema, supported tasks).

#### Data Model Example:

```json
{
  "tool_id": "search_web_v1",
  "name": "Web Search",
  "type": "LangChainTool",
  "capabilities": ["search", "fetch_content"],
  "input_schema": { "query": "string" },
  "output_schema": { "results": "list" },
  "dependencies": ["http_client_v1"],
  "status": "active"
}
```

---

### 2.2 Dependency Graph Manager

Manages the **execution DAG** for tools.

#### Features:

* **Static configuration**: Predefined dependency graphs stored in config files.
* **Dynamic configuration**: Modify or create DAG nodes/edges at runtime via API.
* **Dependency resolution**: Ensure prerequisites run before dependent tools.
* **Graph versioning**: Keep track of historical DAG changes for reproducibility.
* **Graph querying API**: Allow Orchestrix to request partial or full execution graphs.

#### Example DAG:

```
[Data Fetch Tool] → [Data Cleaning Tool] → [Data Analysis Tool] → [Visualization Tool]
```

---

### 2.3 Orchestration Engine

Responsible for mapping incoming **tasks** to the correct tools and execution paths.

#### Workflow:

1. Receive **execution request** from Orchestrix.
2. Lookup required tools & dependencies in the Registry.
3. Build **execution plan** using the Dependency Graph.
4. Dispatch execution to the **Execution Interface**.

#### Selection Modes:

* **Direct Mapping**: Known tool for known task.
* **Capability Matching**: Match required task capability to available tools.
* **Fallback Chain**: Try primary → secondary → backup tools.

---

### 2.4 Execution Interface

Provides unified method for running tools regardless of implementation type.

#### Supported Tool Types:

* **LangChain Tool** — Invoked via LangChain's `Tool` interface.
* **MCP Tool** — Invoked via MCP protocol.
* **Custom REST API** — Invoked via HTTP requests.
* **Local Script/Command** — Invoked via subprocess execution.

#### Execution Flow:

```plaintext
Execution Request → Toolify API → Tool Adapter → Execute Tool → Return Output
```

#### Adapters:

* **LangChainAdapter**
* **MCPAdapter**
* **HTTPAdapter**
* **LocalAdapter**

---

### 2.5 API Layer

Expose HTTP/WebSocket interface for internal and external services.

#### API Endpoints:

* `POST /tools/register` — Register new tool.
* `GET /tools` — List available tools.
* `POST /execute` — Execute a tool by ID or capability.
* `POST /graph/update` — Modify DAG.
* `GET /graph` — Retrieve DAG.

---

### 2.6 Monitoring & Logging

Ensures observability and reliability.

#### Metrics:

* Tool invocation count.
* Execution latency per tool.
* Failure rate & error types.

#### Logging:

* Structured logs (JSON).
* Persistent history of all executions.
* Audit trail for security compliance.

---

## 3. Example Execution Flow

### Example: "Summarize a Web Page"

1. **Orchestrix** detects goal: "Summarize a webpage".
2. Sends request to Toolify:

```json
{
  "goal": "summarize_webpage",
  "params": { "url": "https://example.com" }
}
```

3. Toolify looks up:

   * **Tool A**: `web_fetcher_v1`
   * **Tool B**: `text_cleaner_v2`
   * **Tool C**: `summarizer_llm_v3`
4. Builds execution DAG:

```
web_fetcher_v1 → text_cleaner_v2 → summarizer_llm_v3
```

5. Executes sequentially via Execution Interface.
6. Returns final summary to Orchestrix.

---

## 4. Technology Choices

| Component            | Suggested Tech                |
| -------------------- | ----------------------------- |
| Registry & Metadata  | PostgreSQL / SQLite           |
| API Layer            | FastAPI / Flask               |
| Dependency Graph     | NetworkX / custom DAG manager |
| Execution Interface  | Python adapters + asyncio     |
| Logging & Monitoring | OpenTelemetry + Prometheus    |

---

## 5. Future Extensions

* **Tool version rollback** — revert to older version of tool execution logic.
* **Capability-based marketplace** — automatic discovery of new tools from external sources.
* **Self-healing graph** — detect broken dependencies and auto-remap.
* **Caching layer** — store frequent tool outputs for speed.
