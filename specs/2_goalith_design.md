# Goalith: Current Design Summary

## Overview
Goalith is a DAG-based goal and to-do management system focused on hierarchical goal modeling, flexible decomposition, reactive updates, and extensibility. It manages goals, subgoals, and tasks with explicit dependency and status tracking but does not handle execution or agent interaction directly.

---

## Core Functionalities

### 1. DAG-Based Goal Modeling
- Represents goals, subgoals, and tasks as nodes in a Directed Acyclic Graph.
- Each node contains metadata: description, status, type, dependencies, children, priority, context, and timestamps.
- Enforces execution order through dependency resolution.

### 2. Goal Decomposition Interface
- Supports pluggable **GoalDecomposer** components to break down goals into subgoals or tasks.
- Decomposers can be humans, AI agents (LLMs), symbolic planners, or any callable entity.
- Decomposition expands the DAG dynamically.

### 3. Reactive Update Mechanism
- Supports asynchronous, event-driven updates for goal status, context, and structure.
- Humans or agents can post updates during execution.
- Notifies subscribers for real-time monitoring and collaboration.

### 4. Prioritization & Scheduling
- Allows assignment of priorities to nodes.
- Supports scheduling policies to select among executable tasks based on priority and other constraints.

### 5. Conflict Detection & Resolution Interface
- Provides an abstract interface to detect and resolve conflicts from concurrent updates.
- Supports integration with LLM reasoning, human input, or symbolic conflict handlers.

### 6. Adaptive Planning & Replanning
- Enables dynamic replanning triggered by task failures, new goals, or changed priorities.
- Exposes hooks for re-decomposition and modification of the goal DAG.

### 7. Contextual Memory Integration
- Defines a memory interface to enrich goal context with external knowledge bases.
- Facilitates multi-agent collaboration and enhanced reasoning.

---

## Responsibilities & Scope
- Purely focused on goal/task modeling, decomposition, tracking, and updates.
- Leaves execution, agent interaction, retry/fallback policies, and orchestration to higher-level systems (e.g., Orchestrix).

---

## Naming & Roles
- **Goalith**: The current goal & to-do manager focused on planning and tracking.
- **Orchestrix**: Reserved for future use as a full multi-agent orchestration and execution system built on top of Goalith.

---

This design positions Goalith as a robust, flexible, and extensible foundation for complex goal management in multi-agent or human-AI collaborative workflows.

---

# Goalith: Component Sketch

Below is a high-level breakdown of the key modules and their responsibilities. Everything is organized so you can swap implementations (e.g. networkx vs. another graph lib) without touching the rest of the system.

---

## 1. Core Graph Module  
**Purpose:** Underlying DAG storage and basic graph operations  
- **GraphStore**  
  - Wraps a `networkx.DiGraph` (or similar)  
  - CRUD on nodes and edges  
  - Query “ready” nodes (all dependencies done)  
  - Persist/load graph snapshots  

- **GoalNode**  
  - Data model for nodes (metadata only)  
  - Lightweight DTO that GraphStore knows how to serialize  

---

## 2. Decomposition Module  
**Purpose:** Expand goals into subgoals/tasks  
- **DecomposerRegistry**  
  - Register/unregister `GoalDecomposer` implementations by name  
  - Lookup and invoke the right decomposer for a node  

- **GoalDecomposer (interface)**  
  - Abstract contract: `decompose(goalNode) → [GoalNode,…]`  
  - Plug in humans, LLM agents, symbolic planners, callables  

---

## 3. Scheduling & Prioritization Module  
**Purpose:** Decide which ready node to surface next  
- **PriorityPolicy**  
  - Encapsulates numeric or categorical priority rules  
  - Compare two ready nodes and pick one  

- **Scheduler**  
  - Given a list of ready nodes, applies `PriorityPolicy`  
  - Exposes methods like `get_next()` and `peek_all()`  

---

## 4. Update & Notification Module  
**Purpose:** Handle external updates and broadcast changes  
- **UpdateQueue**  
  - Thread- or async-safe queue of update events  
  - Standard event schema (status change, edit, add/remove)  

- **UpdateProcessor**  
  - Consumes events, applies them to `GraphStore`  
  - Hooks into Conflict Module before final commit  

- **Notifier**  
  - Observer pattern for subscribers  
  - Emits events when nodes change (status, priority, structure)  

---

## 5. Conflict Management Module  
**Purpose:** Detect and resolve concurrent or semantic conflicts  
- **ConflictDetector**  
  - Watches for illegal states (e.g. cycles, double-booked priorities)  
  - Emits conflict reports  

- **ConflictResolver (interface)**  
  - Abstract contract: `resolve(conflict) → resolutionAction`  
  - Plug in LLM-based reasoning, human adjudication, or rules  

- **ConflictOrchestrator**  
  - Coordinates detection → resolution → application  

---

## 6. Replanning Module  
**Purpose:** Reactively adjust plans when things go awry  
- **ReplanTrigger**  
  - Defines conditions (failure, deadline miss, external signal)  
  - Emits “replan needed” events  

- **Replanner**  
  - Hooks into Decomposition and Scheduling modules  
  - Can re-decompose affected subgraphs or adjust priorities  

---

## 7. Memory Integration Module  
**Purpose:** Enrich goals with external context/history  
- **MemoryInterface (abstract)**  
  - Contract for read/write of contextual data  
  - Could be vector DB, SQL, in-memory cache, etc.  

- **MemoryManager**  
  - Retrieves stored context on node creation or lookup  
  - Persists annotations, execution notes, performance metrics  

---

## 8. API & Integration Layer  
**Purpose:** Expose Goalith functionality to UIs or Orchestrix  
- **Goalith**  
  - Façade aggregating all sub-modules  
  - High-level methods: createGoal, decomposeGoal, nextTask, postUpdate, subscribe, triggerReplan  

- **Adapters**  
  - LangGraph adapter: wrap `Goalith` calls as graph nodes  
  - HTTP/CLI adapter: expose REST endpoints or CLI commands  

---

## Wiring It Together  
1. **Initialization**  
   - Instantiate GraphStore, DecomposerRegistry, Scheduler, etc.  
   - Register built-in decomposers and policies.  

2. **Plan Creation**  
   - Client calls `Goalith.createGoal(...)`  
   - GraphStore adds root node  

3. **Decomposition**  
   - Client or agent invokes `Goalith.decomposeGoal(goalId, decomposerName)`  
   - DecomposerRegistry finds and runs the decomposer → new nodes in GraphStore  

4. **Execution Readiness**  
   - `Scheduler.get_next()` returns the highest-priority ready node  

5. **Reactive Updates & Replanning**  
   - External actors post updates via `Goalith.postUpdate(...)`  
   - UpdateProcessor → ConflictOrchestrator → GraphStore → Notifier  
   - If a ReplanTrigger fires, Replanner adjusts the plan  

6. **Memory Hooks & Notifications**  
   - MemoryManager enriches nodes on demand  
   - Notifier streams events to subscribers (UI, Orchestrix, logs)  

---

This modular sketch ensures each concern is isolated and easily replaceable, while the `Goalith` façade ties everything into a coherent system. Let me know what you’d like to refine next!
