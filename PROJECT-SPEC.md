# AI Agent Harness — Complete Project Specification

## 1. Objective

Build a production-oriented open-source AI Agent Harness using:

* Python 3.12+
* FastAPI
* LangGraph
* Ollama
* Gemma 4
* Pydantic v2
* PostgreSQL
* Redis
* pgvector
* SSE streaming
* pytest
* Docker Compose

The system must support:

1. Agent loop engineering
2. Tool calling
3. Short-term conversation memory
4. Long-term semantic memory
5. Episodic memory
6. Agent planning and replanning
7. Tool retry policies
8. Maximum step limits
9. Timeout limits
10. Tool-call limits
11. Human-in-the-loop approval
12. Evaluation framework
13. Agent trajectory evaluation
14. Observability
15. Structured logging
16. FastAPI REST API
17. FastAPI SSE streaming
18. Docker Compose
19. Unit tests
20. Integration tests
21. Evaluation/regression tests

---

# 2. Architecture

Implement this architecture:

Client
|
v
FastAPI
|
v
Agent Harness
|
+---- Policy Engine
|
+---- Memory Manager
|
+---- LangGraph
|
v
Agent
|
+-----+------+
|            |
Tool Call     Final
|
v
Tool
|
v
Observation
|
v
Agent
|
v
Evaluation

External infrastructure:

PostgreSQL
|
+-- agent state
+-- conversations
+-- memories
+-- evaluations
+-- audit records

pgvector
|
+-- semantic memory

Redis
|
+-- cache
+-- rate limiting
+-- event/pub-sub support

Ollama
|
+-- Gemma 4

---

# 3. Repository Structure

Create this structure:

ai-agent-harness/

├── app/
│   ├── **init**.py
│   ├── main.py
│   │
│   ├── api/
│   │   ├── **init**.py
│   │   ├── routes_agent.py
│   │   ├── routes_memory.py
│   │   ├── routes_health.py
│   │   └── dependencies.py
│   │
│   ├── agent/
│   │   ├── **init**.py
│   │   ├── graph.py
│   │   ├── state.py
│   │   ├── nodes.py
│   │   ├── routing.py
│   │   ├── policies.py
│   │   ├── prompts.py
│   │   └── executor.py
│   │
│   ├── llm/
│   │   ├── **init**.py
│   │   ├── ollama_client.py
│   │   └── model.py
│   │
│   ├── tools/
│   │   ├── **init**.py
│   │   ├── registry.py
│   │   ├── calculator.py
│   │   ├── datetime_tool.py
│   │   └── weather.py
│   │
│   ├── memory/
│   │   ├── **init**.py
│   │   ├── manager.py
│   │   ├── short_term.py
│   │   ├── semantic.py
│   │   ├── episodic.py
│   │   └── summarizer.py
│   │
│   ├── db/
│   │   ├── **init**.py
│   │   ├── database.py
│   │   ├── models.py
│   │   └── repositories.py
│   │
│   ├── evaluation/
│   │   ├── **init**.py
│   │   ├── evaluator.py
│   │   ├── metrics.py
│   │   ├── datasets.py
│   │   └── runner.py
│   │
│   ├── observability/
│   │   ├── **init**.py
│   │   ├── logging.py
│   │   ├── tracing.py
│   │   └── events.py
│   │
│   ├── schemas/
│   │   ├── **init**.py
│   │   ├── agent.py
│   │   ├── memory.py
│   │   └── evaluation.py
│   │
│   └── core/
│       ├── **init**.py
│       ├── config.py
│       ├── exceptions.py
│       └── security.py
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── evaluation/
│
├── evals/
│   └── agent_cases.json
│
├── scripts/
│   ├── seed_data.py
│   └── run_evals.py
│
├── docker/
│   └── Dockerfile
│
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── .gitignore
├── Makefile
├── README.md
└── PROJECT_SPEC.md

---

# 4. Configuration

Use environment variables.

Required variables:

OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gemma4:e4b

POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=agentdb
POSTGRES_USER=agent
POSTGRES_PASSWORD=agent

REDIS_URL=redis://localhost:6379/0

MAX_AGENT_STEPS=10
MAX_TOOL_CALLS=5
MAX_RETRIES=2
AGENT_TIMEOUT_SECONDS=120

Do not hard-code secrets.

Use Pydantic Settings.

---

# 5. Ollama Integration

Create an abstraction:

LLMClient

with:

async chat(...)
async stream(...)
async structured_output(...)

Implement OllamaLLM.

The default model must come from:

OLLAMA_MODEL

Do not hard-code the model throughout the project.

The default development model is:

gemma4:e4b

---

# 6. Agent State

Create a strongly typed AgentState.

It should contain:

* thread_id
* user_id
* messages
* current_plan
* current_step
* max_steps
* tool_calls
* observations
* retrieved_memories
* pending_approval
* final_answer
* status
* error
* metadata

Use TypedDict or an appropriate LangGraph state schema.

---

# 7. Agent Loop

Implement:

START
|
v
LOAD_MEMORY
|
v
AGENT
|
+---- final ----> SAVE_MEMORY ---> END
|
+---- tool -----> TOOL
|
v
OBSERVE
|
v
AGENT

The harness must enforce:

MAX_AGENT_STEPS

MAX_TOOL_CALLS

MAX_RETRIES

AGENT_TIMEOUT_SECONDS

The LLM must never be allowed to override these values.

---

# 8. Tool Calling

Create a tool registry.

Every tool must have:

* name
* description
* input schema
* permission
* timeout
* retry policy
* implementation

Start with:

1. calculator
2. datetime
3. weather

Use Pydantic models for tool inputs.

Do not use unrestricted eval().

The calculator must safely parse mathematical expressions.

---

# 9. Tool Execution Policy

Before executing a tool:

1. Validate tool name.
2. Validate arguments.
3. Validate permission.
4. Check timeout.
5. Check tool-call budget.
6. Execute.
7. Capture result.
8. Capture latency.
9. Capture errors.
10. Add observation to AgentState.

---

# 10. Retry Policy

Implement retry behavior.

Retry only transient failures.

Do not retry:

* invalid arguments
* permission failures
* authentication failures
* permanent business errors

Record:

attempt
tool
error
latency
final status

---

# 11. Memory

Implement three memory types.

## Short-term memory

Current conversation/thread state.

Use LangGraph checkpointing.

## Episodic memory

Store previous interactions:

* task
* action
* result
* success
* timestamp

## Semantic memory

Store durable user knowledge.

Example:

User prefers Python.

Use PostgreSQL + pgvector for semantic memory.

Memory retrieval must be relevance-based.

Do not blindly inject all memories into prompts.

---

# 12. Memory Manager

Create:

MemoryManager

Methods:

retrieve_relevant_memories()

save_semantic_memory()

save_episode()

summarize_conversation()

delete_memory()

list_memories()

The agent should retrieve only the memories relevant to the current request.

---

# 13. Prompt Architecture

Do not create one huge prompt.

Use:

SYSTEM_PROMPT

MEMORY_CONTEXT

TASK_CONTEXT

TOOL_CONTEXT

AGENT_INSTRUCTIONS

CURRENT_CONVERSATION

The final prompt should be assembled dynamically.

---

# 14. Safety / Policy Engine

Create PolicyEngine.

It must control:

* maximum steps
* maximum tool calls
* timeouts
* allowed tools
* human approval
* retry limits

The LLM must not bypass the policy engine.

---

# 15. Human Approval

Implement interrupt/resume behavior.

High-risk tools must require approval.

Example:

Agent
|
v
"Send email"
|
v
Policy Engine
|
v
Approval Required
|
v
FastAPI
|
v
Human
|
+---- approve
|
+---- reject

Implement:

POST /agent/{thread_id}/approve

POST /agent/{thread_id}/reject

---

# 16. FastAPI API

Implement:

POST /agent/run

POST /agent/stream

GET /agent/{thread_id}

POST /agent/{thread_id}/approve

POST /agent/{thread_id}/reject

GET /memory

POST /memory

DELETE /memory/{memory_id}

GET /health

GET /ready

GET /evaluations

Use Pydantic request/response schemas.

---

# 17. SSE Streaming

Implement:

POST /agent/stream

Stream structured events.

Event types:

agent_start

agent_step

llm_start

llm_token

llm_end

tool_start

tool_result

memory_retrieved

approval_required

agent_error

agent_complete

The client should be able to reconstruct the complete agent trajectory from these events.

---

# 18. Observability

Every agent run needs:

run_id

thread_id

user_id

timestamp

step

event

node

tool

latency

success

error

metadata

Use structured JSON logging.

Create an event abstraction so OpenTelemetry can be added later.

---

# 19. Evaluation Framework

Create evaluation cases:

{
"id": "case_001",
"input": "...",
"expected": "...",
"tools": [],
"criteria": []
}

Implement:

TaskSuccessEvaluator

AnswerQualityEvaluator

ToolSelectionEvaluator

TrajectoryEvaluator

MemoryEvaluator

EfficiencyEvaluator

---

# 20. Evaluation Metrics

Track:

task_success

answer_quality

tool_accuracy

tool_success

trajectory_quality

memory_precision

memory_recall

steps

tool_calls

latency_ms

retry_count

errors

Create an aggregate score.

---

# 21. Regression Testing

Create at least 20 evaluation cases.

Running:

python scripts/run_evals.py

must produce a report such as:

Total cases: 20
Passed: 18
Failed: 2
Success rate: 90%
Average latency: ...
Average steps: ...
Average tool calls: ...

Fail the process if success rate is below a configurable threshold.

---

# 22. Tests

Create unit tests for:

* configuration
* tool registry
* calculator
* tool validation
* policies
* retry logic
* memory manager
* prompt builder
* evaluator

Create integration tests for:

* FastAPI
* Ollama
* LangGraph
* PostgreSQL
* Redis

Tests requiring external services should be marked appropriately.

---

# 23. Docker Compose

Create services:

api
postgres
redis
ollama

PostgreSQL must support pgvector.

Persist volumes.

Add health checks.

The API must wait for dependencies.

Do not run Ollama inside the API container.

---

# 24. Redis

Use Redis for:

* rate limiting
* caching
* event coordination

Do not use Redis as the permanent source of truth for conversation memory.

---

# 25. Database

Create models for:

users

threads

messages

agent_runs

agent_events

memories

episodes

evaluations

tool_calls

audit_logs

Use SQLAlchemy 2.x async APIs.

Use Alembic migrations.

Do not create database tables automatically at application startup in production.

---

# 26. Security

Implement:

* request validation
* tool authorization
* API authentication abstraction
* rate limiting
* request IDs
* safe error messages
* no secrets in logs
* no arbitrary Python execution
* tool timeout
* maximum request size

---

# 27. Error Handling

Use custom exceptions:

AgentError

LLMError

ToolError

ToolTimeoutError

PolicyViolationError

MemoryError

EvaluationError

Create FastAPI exception handlers.

Return structured errors.

---

# 28. README

README must contain:

1. Architecture
2. Prerequisites
3. Ollama installation
4. Gemma setup
5. Python setup
6. Environment variables
7. Docker setup
8. Running locally
9. API examples
10. Agent loop explanation
11. Memory explanation
12. Evaluation explanation
13. Testing
14. Troubleshooting

---

# 29. Development Rules

Before modifying files:

1. Inspect the repository.
2. Inspect existing files.
3. Do not overwrite working code unnecessarily.
4. Follow existing patterns.
5. Keep modules small.
6. Use type hints.
7. Use async APIs where appropriate.
8. Add tests with every major feature.
9. Run tests after implementation.
10. Fix failures before continuing.

Do not create fake implementations merely to satisfy imports.

Do not silently skip functionality.

If a dependency or external service is unavailable, create a clean abstraction and document the limitation.

---

# 30. Implementation Order

Implement in this exact order:

PHASE 1
Project skeleton
Configuration
Logging
Health API

PHASE 2
Ollama client
Gemma integration
Basic LLM test

PHASE 3
AgentState
LangGraph
Basic agent loop

PHASE 4
Tool registry
Calculator
Datetime
Weather

PHASE 5
Loop policies
Step limits
Tool limits
Timeouts
Retries

PHASE 6
Short-term memory

PHASE 7
PostgreSQL
Long-term memory
pgvector

PHASE 8
FastAPI agent endpoints

PHASE 9
SSE streaming

PHASE 10
Human approval

PHASE 11
Evaluation framework

PHASE 12
Regression tests

PHASE 13
Redis

PHASE 14
Docker Compose

PHASE 15
Security hardening

PHASE 16
Documentation

---

# 31. Acceptance Criteria

The project is complete only when all of the following work.

## Basic agent

User:

"What is 25 * 4?"

Agent:

1. recognizes calculator is appropriate
2. calls calculator
3. receives 100
4. returns final answer

## Multi-step agent

User:

"Find the weather in New York and tell me whether I need an umbrella."

Agent must:

1. call weather tool
2. inspect result
3. reason over result
4. return recommendation

## Memory

User:

"My preferred programming language is Python."

Later:

"What programming language do I prefer?"

Agent should retrieve the memory.

## Loop protection

A pathological task must stop after MAX_AGENT_STEPS.

## Tool protection

The agent must stop after MAX_TOOL_CALLS.

## Evaluation

The evaluation suite must run successfully.

## Streaming

The SSE endpoint must stream agent events.

## Docker

The entire stack must start using:

docker compose up

---

# 32. Final Quality Requirement

Do not stop after creating files.

You must:

1. install dependencies
2. run static checks
3. run unit tests
4. run integration tests where services are available
5. start FastAPI
6. test health endpoint
7. test Ollama
8. test one tool
9. test the complete agent
10. test memory
11. test SSE
12. run evaluation suite
13. fix failures
14. update README

At the end provide:

* files created
* architecture summary
* commands to run
* tests executed
* known limitations

Do not claim the project is complete unless the acceptance criteria have been verified.
