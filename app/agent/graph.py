from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agent.nodes import AgentNode, MemoryNodes, ToolNode, await_approval, observe
from app.agent.policies import PolicyEngine
from app.agent.routing import after_agent, after_approval, after_memory_load
from app.agent.state import AgentContext, AgentState
from app.llm.model import LLMClient
from app.memory.manager import MemoryManager
from app.tools.registry import ToolRegistry


def build_graph(
    llm: LLMClient, policies: PolicyEngine, registry: ToolRegistry | None,
    checkpointer: BaseCheckpointSaver, memory: MemoryManager | None = None,
) -> CompiledStateGraph:
    builder = StateGraph(AgentState, context_schema=AgentContext)
    builder.add_node("agent", AgentNode(llm, policies, registry))
    builder.add_node("approval", await_approval)
    builder.add_node("tool", ToolNode(policies))
    builder.add_node("observe", observe)
    memory_nodes = MemoryNodes(memory)
    builder.add_node("load_memory", memory_nodes.load)
    builder.add_node("save_memory", memory_nodes.save)
    builder.add_edge(START, "load_memory")
    builder.add_conditional_edges(
        "load_memory", after_memory_load, {"agent": "agent", END: END}
    )
    builder.add_conditional_edges(
        "agent", after_agent, {"tool": "tool", "approval": "approval", "save_memory": "save_memory"}
    )
    builder.add_conditional_edges(
        "approval", after_approval, {"tool": "tool", "observe": "observe"}
    )
    builder.add_edge("save_memory", END)
    builder.add_edge("tool", "observe")
    builder.add_edge("observe", "agent")
    return builder.compile(checkpointer=checkpointer)
