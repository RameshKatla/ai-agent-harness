# AI Agent Harness

This project will expose an agent through FastAPI and use LangGraph to manage
its execution, Ollama to run Gemma, PostgreSQL with pgvector to store memory,
and Redis for rate limiting, caching, and event coordination.

## Phase 1 setup

Use Python 3.12 or later.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
make run
```

The API reads `.env` from the working directory. Environment variables take
precedence. See [.env.example](.env.example) for the settings and development
values. Set your own PostgreSQL credentials when configuring a database.

```sh
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8000/ready
make check
make test
```

`/health` checks HTTP liveness. In Phase 1, `/ready` reports whether the API
can serve requests. Dependency checks will be added as services are wired in.
The application emits JSON logs to standard error.

## Phase 2 local Ollama

Install Ollama and start it with `ollama serve` if it isn't already running.
Use an installed model tag for `OLLAMA_MODEL`. The specification's default is
`gemma4:4b`. Ollama publishes the 4B variant as `gemma4:e4b`
([Gemma 4 tags](https://ollama.com/library/gemma4/tags)).

```sh
ollama pull gemma4:e4b
RUN_OLLAMA_TESTS=1 OLLAMA_MODEL=gemma4:e4b python -m pytest -m external
```

The client supports chat, incremental text streaming, and JSON schema output
validated with Pydantic. The normal test suite exercises all three paths using
an in-process HTTP transport. Set `OLLAMA_BASE_URL` if the live server uses
a different address. If a request fails, check that the server is running
and `ollama list` includes the configured model.

## Phase 3 basic agent

`AgentExecutor.run()` initializes typed state and invokes the LangGraph agent
node. The node sends the task to the injected model and records either a final
answer and assistant message or a failed run with the model error. The graph
runs a single model step when no tool registry is supplied.

The opt-in Ollama suite also checks the complete basic graph with a live model.

## Phase 4 built-in tools

`build_tool_registry(http)` registers three tools. The caller supplies its
permissions to `ToolRegistry.execute()`. Each tool has a Pydantic input
schema, a permission, a deadline, and a retry policy.

| Tool | Example arguments | Permission |
| --- | --- | --- |
| Calculator | `{"expression": "25 * 4"}` | `calculator:use` |
| Datetime | `{"timezone": "America/New_York"}` | `datetime:use` |
| Weather | `{"location": "New York, US"}` | `weather:use` |

The datetime tool uses the IANA timezone database. Minimal Unix images need
system timezone data installed. Weather uses Open-Meteo's public geocoding and
forecast endpoints and requires outbound HTTPS. It returns the best matching
location, current conditions, and today's maximum precipitation probability.
See [Open-Meteo](https://open-meteo.com/en/docs) for data and service terms.


## Phase 5 tool-enabled agent

Pass `build_tool_registry(http)` as the third argument to `AgentExecutor`
and supply the caller's tool permissions to `run()`. LangGraph repeats
model decision, tool execution, and observation until the model answers
or a harness limit stops the run. The model can propose and revise a plan
through structured decisions. The generation schema includes only permitted
tools and their input schemas. The harness validates every proposed call.

`MAX_AGENT_STEPS` caps model decisions. `MAX_TOOL_CALLS` caps tool proposals
admitted for execution. A model can finalize after its last allowed tool call.
Per-tool retries share that call's budget slot and are capped by both
`MAX_RETRIES` and the tool's retry policy. Only transient failures and tool
timeouts can retry. `AGENT_TIMEOUT_SECONDS` covers the entire run. Tool
observations include the outcome, error, and latency of each completed attempt.

The opt-in Ollama tests include an end-to-end calculator call.

## Phase 6 short-term conversation memory

Reuse the same `AgentExecutor` for follow-ups with the same user and thread
IDs. LangGraph checkpoints keep the completed user and assistant exchanges.
Each turn starts with a fresh policy budget, and tool call transcripts stay
within their original turn. Failed and timed-out turns are omitted from the
history given to the model. `MAX_CONVERSATION_TURNS` limits how many previous
exchanges are provided to each turn and defaults to 20.

The checkpoint store is process-local and lasts for the lifetime of the
executor. Restarting or replacing the executor clears this history. The current
in-memory store retains step checkpoints as well as the bounded prompt history.

## Phase 7 durable memory

Use PostgreSQL with the pgvector extension available. Set the `POSTGRES_*`
variables from `.env.example` for a database the migration role can manage.
Install the embedding model and apply migrations before using durable memory.

```sh
ollama pull nomic-embed-text
python -m alembic upgrade head
RUN_POSTGRES_TESTS=1 python -m pytest tests/integration/test_postgres_memory.py
RUN_POSTGRES_TESTS=1 RUN_OLLAMA_TESTS=1 OLLAMA_MODEL=gemma4:e4b python -m pytest -m external
```

Configure `OLLAMA_EMBEDDING_MODEL` with a model that supports 768-dimensional
embeddings. Search compares vectors from the same model. Changing models
requires re-embedding existing entries before the new model can retrieve them.
`MEMORY_RETRIEVAL_LIMIT` and `MEMORY_MIN_SIMILARITY` control the number and
minimum cosine similarity of the user-scoped results included in the prompt.

In Python, create an engine with `create_database_engine(settings)`, build
a `MemoryManager.using_database(settings, engine, embedder, llm)`, and pass it
as `memory=` to `AgentExecutor`. Dispose of the engine when finished. A
memory-enabled run saves an episode when it reaches the save step. It saves a
durable preference automatically for standalone declarations such as
“My preferred programming language is Python.” A newer declaration for
the same subject replaces the previous value. Explicit semantic saves are
also available via `MemoryManager.save_semantic_memory()`. Short-term
checkpoints remain process-local. HTTP routes for agent runs and memory
management arrive in Phase 8.

## Phase 8 REST API

Apply the database migrations and start local Ollama with the configured chat
and embedding models before calling the API. Set `API_TOKEN` to a private
bearer token and `API_USER_ID` to its stable owner identifier. Phase 8 maps
one configured token to one principal per deployment. Requests derive
identity and tool permissions from that principal. The health endpoints
remain public.

```sh
python -m alembic upgrade head
make run
```

From another shell with the same `API_TOKEN` exported:

```sh
curl -fsS http://127.0.0.1:8000/agent/run \
  -H "Authorization: Bearer $API_TOKEN" -H 'Content-Type: application/json' \
  -d '{"thread_id":"examples","task":"What is 25 * 4?"}'

curl -fsS http://127.0.0.1:8000/agent/examples \
  -H "Authorization: Bearer $API_TOKEN"

curl -fsS http://127.0.0.1:8000/memory \
  -H "Authorization: Bearer $API_TOKEN" -H 'Content-Type: application/json' \
  -d '{"content":"My preferred programming language is Python."}'

curl -fsS 'http://127.0.0.1:8000/memory?limit=20&offset=0' \
  -H "Authorization: Bearer $API_TOKEN"

curl -fsS -X DELETE "http://127.0.0.1:8000/memory/$MEMORY_ID" \
  -H "Authorization: Bearer $API_TOKEN"
```

Set `MEMORY_ID` to the ID returned by the memory create request. An omitted
`thread_id` in an agent run starts a new thread. Agent runs return HTTP 200
with their execution status, including `failed` for a policy or model failure.
The thread lookup returns the latest stored run, which remains readable after
an API restart. Conversation context for new turns uses the process-local
checkpoint.
## Phase 9 server-sent events

Use the same request body and bearer token as `POST /agent/run`.

```sh
curl -fsS -N http://127.0.0.1:8000/agent/stream \
  -H "Authorization: Bearer $API_TOKEN" -H 'Content-Type: application/json' \
  -d '{"thread_id":"stream-example","task":"What is 25 * 4?"}'
```

Each SSE frame has an `event` name, an ID in the form `run_id:sequence`, and
one JSON `data` object. The payload identifies the run, user, thread, node,
step, and timestamp. The stream reports memory retrieval, model steps, and
tool calls and outcomes. `llm_token` contains a model output fragment.
Concatenate fragments for each step to see the generated decision. For
tool-enabled runs these fragments are structured JSON, which the harness
validates before using. Read the user-facing answer from the validated
`agent_complete.metadata.run.final_answer`.

An execution failure emits `agent_error` followed by `agent_complete` with
a failed status. When persistence succeeds, the terminal event includes the
same run returned by the thread lookup. A persistence failure yields a
terminal failure without that run object. Keep-alive comments can appear
during long pauses. Disconnecting cancels unfinished work. Event IDs order
the live stream. Reconnecting starts a new request, and event replay is
currently unavailable.


## Phase 10 human approval

A registered tool can be marked `requires_approval=True`. Operators can also
gate existing tools through `APPROVAL_REQUIRED_TOOLS`, a JSON array of tool
names. The built-in read-only tools do not require approval by default. To
exercise the gate locally, start the API with this override.

```sh
APPROVAL_REQUIRED_TOOLS='["calculator"]' make run
```

When a run pauses, its status is `awaiting_approval` and `pending_approval`
contains the tool name, validated arguments, and a unique `approval_id`.
A streamed run ends its connection with `approval_required`, including the
paused run under `metadata.run`. Submit the ID to either decision endpoint.

```sh
curl -fsS -X POST "http://127.0.0.1:8000/agent/$THREAD_ID/approve" \
  -H "Authorization: Bearer $API_TOKEN" -H 'Content-Type: application/json' \
  -d "{\"approval_id\":\"$APPROVAL_ID\"}"
```

Use `/reject` to resume with a rejection observation. Either route returns
the resulting JSON run with the same run ID. A duplicate, stale, or mismatched
decision returns HTTP 409. A new run on the same thread also returns HTTP 409
while an approval is pending. Tool permissions are checked again at execution,
and time spent waiting for a person does not consume the active run budget.

The API persists the run record in PostgreSQL. Pending approvals can be
resumed while the issuing API process is running. They use its in-memory
LangGraph checkpoint, so a restart or routing to a different worker prevents
resumption. Start a fresh run after a restart.

## Phase 11 evaluation

Run the regression dataset with an installed Ollama model.

```sh
OLLAMA_MODEL=gemma4:e4b python scripts/run_evals.py
```

Each JSON case specifies the task, an expected answer phrase, ordered tool
calls, and optional required answer phrases, case sensitivity, expected memories,
and efficiency limits. A case passes when the answer, trajectory, tools, memory,
and each configured limit meet the case expectations. A case can also require
the answer to include a value from a successful tool observation. The
reported aggregate averages the component scores and is independent of the
pass rate. Use `--json` for per-case metrics and `--min-success-rate 0.9` to
set a process exit threshold. The default threshold is 100%.

Use `--persist` with a migrated PostgreSQL database to save the evaluated runs
and their scores. Run under a dedicated `API_USER_ID` to isolate evaluation
data from regular interactions. Retrieve that user's results with authenticated
`GET /evaluations?limit=20&offset=0`.

The Phase 12 dataset includes 20 cases covering direct answers, calculator
operations, and timezone lookups. The pytest suite separately checks weather
reasoning against a fixed forecast.
