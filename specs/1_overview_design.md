# Multi-Agent System Overview Design

---

## Components Summary

| Component   | Role & Responsibility                                                                                  |
|-------------|------------------------------------------------------------------------------------------------------|
| **Goalith** | Manages hierarchical goals, subgoals, and tasks as a DAG. Handles decomposition, scheduling, updates, and conflict resolution. Exposes APIs for goal/task management. |
| **Toolify** | Central registry and orchestrator for tools and subagents. Manages tool metadata, dependencies, execution planning, communication, and monitoring. |
| **Orchestrix** | Coordinates workflows by bridging Goalith and Toolify. Fetches executable tasks, requests tool plans, dispatches execution, and handles feedback/replanning. |

---

## Goalith — DAG-Based Goal & Task Management

- **Purpose:** Manage goal/task hierarchy and lifecycle.  
- **Key Modules:**  
  - GraphStore: DAG data structure and operations  
  - GoalNode: Node metadata (description, status, dependencies, priority)  
  - DecomposerRegistry: Pluggable goal decomposers  
  - Scheduler: Prioritization and task selection  
  - UpdateProcessor: Processes updates with conflict detection  
  - ConflictManager: Detects and resolves DAG conflicts  
  - Replanner: Dynamic plan adjustments  
  - MemoryManager: Context enrichment  
  - Notifier: Event notification system  

---

## Toolify — Tool & Subagent Management & Orchestration

- **Purpose:** Manage functional tools and subagents, generate execution plans, facilitate communication and monitoring.  
- **Key Modules:**  
  - Tool Registry: Register/unregister tools with metadata  
  - Metadata Store: Persistent searchable tool data  
  - Dependency Graph: Tool/subagent dependencies (static + dynamic)  
  - Execution Planner: Creates optimized execution plans  
  - Communication Hub: Messaging for coordination  
  - Monitoring & Logging: Tracks execution and reports  

---

## Orchestrix — Multi-Agent Workflow Orchestration & Coordination

- **Purpose:** Bridge Goalith and Toolify to execute workflows end-to-end.  
- **Responsibilities:**  
  - Fetch ready tasks from Goalith  
  - Query Toolify for capable tools  
  - Request execution plans from Toolify  
  - Dispatch execution commands  
  - Aggregate execution feedback  
  - Trigger replanning when needed  
  - Subscribe to event notifications  
- **Interaction:** Acts as central coordinator and communication hub between Goalith and Toolify.

---

## Interaction Flow

1. User or system creates goals/tasks in Goalith.  
2. Orchestrix queries Goalith for ready tasks.  
3. Orchestrix queries Toolify for tools matching tasks.  
4. Toolify generates execution plans for those tasks.  
5. Orchestrix dispatches execution via Toolify to subagents/tools.  
6. Toolify monitors execution, reports back to Orchestrix.  
7. Orchestrix updates Goalith on task status.  
8. Goalith triggers replanning if necessary.  
9. Loop continues until goals complete.

---

## Data Model Snapshots

### Goalith GoalNode Example

```json
{
  "id": "goal_001",
  "description": "Complete research paper",
  "status": "pending",
  "dependencies": ["goal_000"],
  "priority": 10,
  "context": { "deadline": "2025-09-01" }
}
````

### Toolify Tool Metadata Example

```json
{
  "tool_id": "nlp_summarizer_v1",
  "name": "NLP Summarizer",
  "capabilities": ["text_summarization"],
  "input_schema": { "text": "string" },
  "output_schema": { "summary": "string" },
  "version": "1.0"
}
```

### Orchestrix Task Execution Request Example

```json
{
  "task_id": "task_123",
  "tool_id": "nlp_summarizer_v1",
  "input": {
    "text": "Long research paper content..."
  },
  "execution_parameters": {
    "timeout": 120
  }
}
```

---

## Design Considerations

* **Modularity:** Each component is independently extensible and replaceable.
* **Event-Driven:** Use async messaging and notifications for decoupling and scalability.
* **Dynamic Dependency Management:** Both goals and tools support dynamic updates to dependencies.
* **Robust Feedback Loop:** Continuous execution monitoring feeds back to goal/task management for replanning.
* **Security:** Define access control for APIs and communication channels.
