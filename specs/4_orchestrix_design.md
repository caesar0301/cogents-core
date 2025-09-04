# Orchestrix Detailed Design Document

---

## 1. Overview

**Orchestrix** is the **multi-agent orchestration system** designed to bridge and coordinate workflows between **Goalith** and **Toolify**. It fetches executable goals from Goalith, queries Toolify for suitable tools, dispatches execution requests, and handles status feedback. Orchestrix is responsible for managing the entire workflow lifecycle from task fetching to final result aggregation.

---

## 2. Purpose

Orchestrix coordinates tasks across multiple agents and tools in a flexible and scalable way:
- **Task Fetching**: Retrieves executable goals from Goalith.
- **Execution Planning**: Queries Toolify for execution plans.
- **Dispatching Execution**: Sends execution commands to Toolify, invoking tools and subagents.
- **Status Aggregation**: Collects execution feedback and updates Goalith.
- **Replanning**: Triggers replanning based on task failures or updates.

---

## 3. Core Components

### 3.1 Task Fetching & Coordination
- **Goalith Integration**: Queries Goalith for **ready tasks** based on DAG status and dependencies.
- **Task Queueing**: Tasks are placed in a queue based on priority and readiness.
- **Task Scheduling**: Determines task execution order based on predefined priority policies.

#### Example Interaction:
```json
{
  "task_id": "task_123",
  "goal_id": "goal_001",
  "status": "ready_for_execution",
  "dependencies": ["task_122"]
}
````

---

### 3.2 Tool Discovery & Execution Plan Request

* **Toolify Integration**: For each task fetched, Orchestrix queries Toolify to **find suitable tools** for execution, respecting the task's required capabilities.
* **Execution Plan Generation**: Toolify returns an **execution plan** that specifies the sequence of tools/subagents required to complete the task.
* **Dependency Mapping**: Orchestrix ensures dependencies (both tools and tasks) are handled.

#### Example Execution Request to Toolify:

```json
{
  "task_id": "task_123",
  "tools_needed": ["text_cleaner_v1", "summarizer_v2"],
  "execution_order": ["text_cleaner_v1", "summarizer_v2"]
}
```

---

### 3.3 Dispatching Execution

* **Communication Hub**: Orchestrix dispatches the execution to Toolify via the **Communication Hub**, which forwards the execution command to the selected tools or subagents.
* **Subagent Interaction**: Subagents execute the plan, send real-time updates, and return results.
* **Execution Flow**:

  1. Toolify invokes the tool or subagent.
  2. The tool executes the task and returns results.
  3. Toolify monitors execution and logs results.

---

### 3.4 Status Aggregation & Feedback

* **Execution Monitoring**: Orchestrix collects execution feedback from Toolify.

  * **Success or Failure**: Determines whether the execution succeeded or failed.
  * **Task Updates**: Tracks progress and stores intermediate results.
* **Update Goalith**: Orchestrix updates **Goalith** with the current task status (e.g., `completed`, `failed`, `in-progress`).
* **Result Aggregation**: Orchestrix collects the final results from Toolify and aggregates them for higher-level task completion.

#### Example Status Update to Goalith:

```json
{
  "task_id": "task_123",
  "status": "completed",
  "result": {
    "summary": "The research paper discusses...",
    "execution_time": "120s"
  }
}
```

---

### 3.5 Replanning Mechanism

* **Failure Detection**: Orchestrix listens for task execution failures or unexpected status updates.
* **Triggering Replanning**: If a task fails, Orchestrix triggers replanning in Goalith and Toolify, selecting alternate tools or agents, or adjusting priorities.
* **Adaptive Scheduling**: Orchestrix re-adjusts the task queue based on changes in goal/task priorities.

---

### 3.6 Notification & Event Handling

* **Event Subscription**: Orchestrix subscribes to **Goalith** and **Toolify** notifications for updates on task status, failures, or changes.
* **Event-Driven**: Uses a pub/sub model for real-time updates on task execution and changes.

---

## 4. API Layer

Orchestrix exposes several **REST API endpoints** to facilitate interaction with **Goalith**, **Toolify**, and external systems:

#### 4.1 Task Fetching

* **Endpoint**: `GET /tasks/ready`
* **Description**: Fetch ready tasks from Goalith.
* **Response**:

```json
{
  "tasks": [
    {
      "task_id": "task_123",
      "goal_id": "goal_001",
      "status": "ready_for_execution"
    }
  ]
}
```

#### 4.2 Tool Discovery & Plan Generation

* **Endpoint**: `POST /tools/plan`
* **Description**: Generate execution plan for a task based on required tools.
* **Request**:

```json
{
  "task_id": "task_123",
  "tools_needed": ["text_cleaner_v1", "summarizer_v2"]
}
```

* **Response**:

```json
{
  "plan": [
    "text_cleaner_v1",
    "summarizer_v2"
  ]
}
```

#### 4.3 Execution Dispatch

* **Endpoint**: `POST /execution/dispatch`
* **Description**: Dispatch execution to the selected tools.
* **Request**:

```json
{
  "task_id": "task_123",
  "plan": ["text_cleaner_v1", "summarizer_v2"]
}
```

#### 4.4 Status Updates

* **Endpoint**: `POST /tasks/status`
* **Description**: Update the status of a task.
* **Request**:

```json
{
  "task_id": "task_123",
  "status": "completed",
  "result": {
    "summary": "The research paper discusses..."
  }
}
```

#### 4.5 Replanning Trigger

* **Endpoint**: `POST /replan`
* **Description**: Trigger replanning due to failure or new goal/task.
* **Request**:

```json
{
  "task_id": "task_123",
  "reason": "execution failure"
}
```

---

## 5. Technology Stack

| Component        | Suggested Technology               |
| ---------------- | ---------------------------------- |
| API Layer        | FastAPI / Flask / Express          |
| Task Queueing    | Redis / Celery                     |
| Event System     | RabbitMQ / Kafka / WebSockets      |
| Execution Logic  | Python (asyncio, subprocess)       |
| Monitoring       | OpenTelemetry, Prometheus, Grafana |
| Database         | PostgreSQL / SQLite                |
| Dependency Graph | NetworkX / custom graph manager    |

---

## 6. Future Enhancements

* **Self-Healing Orchestration**: Auto-resolve failures and select alternative execution paths.
* **Advanced Scheduling**: Implement resource-based scheduling and load balancing.
* **Distributed Execution**: Extend orchestration to multi-machine, multi-cluster environments.
* **Tool Marketplace**: Automatically discover and integrate new tools from external sources.

---

## 7. Integration with Goalith & Toolify

Orchestrix operates as a **client to Goalith** and **Toolify**, managing the entire lifecycle of task orchestration:

* **Goalith**: Provides ready-to-execute tasks based on dependencies and priority.
* **Toolify**: Supplies tools/subagents for execution, returns execution results, and facilitates monitoring/logging.

Orchestrix maintains a **feedback loop** that continuously updates task status in Goalith and adjusts execution plans based on feedback from Toolify.

---

## 8. Conclusion

Orchestrix serves as the **central coordinator** in the multi-agent system, ensuring that **Goalith's goals** are matched to **Toolify's tools** for efficient, scalable, and adaptable task execution. By using a combination of **task scheduling**, **tool orchestration**, and **real-time event handling**, Orchestrix ensures that the system responds dynamically to changing conditions, supporting complex workflows and seamless integration.
