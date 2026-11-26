from typing import Literal

from app.agent.state import AgentState


def after_agent(state: AgentState) -> Literal["tool", "approval", "save_memory"]:
    if state["status"] == "awaiting_approval":
        return "approval"
    if state["status"] == "running" and state["proposed_tool"] is not None:
        return "tool"
    return "save_memory"


def after_memory_load(state: AgentState) -> Literal["agent", "__end__"]:
    return "agent" if state["status"] == "running" else "__end__"



def after_approval(state: AgentState) -> Literal["tool", "observe"]:
    return "tool" if state["proposed_tool"] is not None else "observe"
