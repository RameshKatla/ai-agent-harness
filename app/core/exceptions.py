class AgentError(Exception):
    """Base error for harness operations."""


class LLMError(AgentError):
    """The model request failed or returned an unusable response."""


class ToolError(AgentError):
    """A tool request could not be completed."""


class ToolInputError(ToolError):
    """The tool name or arguments are invalid."""


class ToolTimeoutError(ToolError):
    """A tool exceeded its execution deadline."""


class TransientToolError(ToolError):
    """A tool failed due to a temporary condition."""


class PolicyViolationError(AgentError):
    """A harness policy prevented an operation."""


class MemoryError(AgentError):
    """A memory operation failed."""


class StorageError(AgentError):
    """A durable record could not be written or read."""


class ApprovalConflictError(AgentError):
    """The requested approval transition conflicts with the current thread state."""


class EvaluationError(AgentError):
    """An evaluation dataset or run could not be processed."""
