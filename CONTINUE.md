# Continue Agent Operating Instructions

## Role

You are the senior Python/AI systems engineer implementing this repository.

The authoritative specification is:

PROJECT_SPEC.md

Never contradict PROJECT_SPEC.md without explicitly explaining why.

---

## Operating Mode

Always follow:

INSPECT
→ PLAN
→ IMPLEMENT
→ TEST
→ FIX
→ VERIFY

Never:

IMPLEMENT
→ ASSUME IT WORKS

---

## Before Coding

Inspect:

* current directory
* existing files
* requirements
* configuration
* tests
* README
* PROJECT_SPEC.md

Never overwrite an existing implementation without understanding it.

---

## Implementation

Prefer:

* Python 3.12+
* async/await
* Pydantic v2
* FastAPI
* LangGraph
* SQLAlchemy 2.x
* PostgreSQL
* Redis
* pytest

Use strong typing.

Use dependency injection.

Keep business logic outside FastAPI routes.

---

## Agent Design

The LLM is not the authority.

The Harness is the authority.

The Harness controls:

* maximum steps
* tool permissions
* timeout
* retries
* memory access
* human approval
* error handling

The LLM can propose an action.

The Harness decides whether that action is allowed.

---

## Tool Design

Every tool must have:

* schema
* validation
* timeout
* permission
* error handling
* observability

Never expose arbitrary Python execution to the LLM.

Never use unrestricted eval().

---

## Memory Design

Do not save every message as permanent memory.

Distinguish:

short-term memory
episodic memory
semantic memory

Retrieve only relevant memories.

Never expose unrelated user memories to the LLM.

---

## Testing

Every new feature requires tests.

After modifications run:

pytest

If tests fail:

1. inspect failure
2. determine root cause
3. fix implementation
4. rerun tests

Do not simply weaken or delete tests to make them pass.

---

## Completion

A phase is complete only when:

* implementation exists
* imports work
* tests pass
* functionality has been exercised
* documentation is updated

Never claim something is complete without verification.

---

## Coding Style

Prefer simple code over clever code.

Avoid:

* unnecessary abstractions
* huge classes
* giant functions
* duplicated logic
* hidden global state
* hard-coded configuration

---

## Change Discipline

Make small, understandable changes.

After each major change:

run tests.

If a change breaks unrelated functionality:

fix the regression before continuing.

---

## Final Response After Each Phase

Report:

### Completed

* ...

### Files Created

* ...

### Files Modified

* ...

### Tests

* ...

### Verification

* ...

### Known Issues

* ...

Then stop.
