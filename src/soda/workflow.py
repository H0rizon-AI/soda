from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, END

from soda.types import AgentOptimizationState, CritiqueOutput, RefinementOutput, TaskAnalysisOutput
from soda.prompts import (
    TASK_ANALYSIS_PROMPT,
    CRITIQUE_PROMPT,
    REFINEMENT_PROMPT,
    format_agents_for_prompt,
    format_improvements_for_prompt
)


def task_analyzer_node(state: AgentOptimizationState) -> AgentOptimizationState:
    """Analyze the task and generate initial sub-agents and orchestration strategy."""

    llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0.7, max_tokens=8192)
    structured_llm = llm.with_structured_output(TaskAnalysisOutput)

    analysis_prompt = TASK_ANALYSIS_PROMPT.format(task=state['task'])

    response = structured_llm.invoke([HumanMessage(content=analysis_prompt)])

    return {
        **state,
        "sub_agents": response.sub_agents,
        "orchestration_strategy": response.orchestration_strategy,
        "iteration": state.get("iteration", 0) + 1
    }


def critic_node(state: AgentOptimizationState) -> AgentOptimizationState:
    """Reflect on and score the generated sub-agents and orchestration strategy."""

    llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0.7, max_tokens=8192)
    structured_llm = llm.with_structured_output(CritiqueOutput)

    # Format sub-agents for the prompt
    agents_text = format_agents_for_prompt(state["sub_agents"])

    critique_prompt = CRITIQUE_PROMPT.format(
        task=state['task'],
        agents_text=agents_text,
        orchestration_strategy=state['orchestration_strategy']
    )

    response = structured_llm.invoke([HumanMessage(content=critique_prompt)])

    return {
        **state,
        "critique": response.critique,
        "score": response.score,
        "improvements": response.improvements
    }


def refinement_node(state: AgentOptimizationState, config: RunnableConfig) -> AgentOptimizationState:
    """Refine the agents and strategy based on critique if score is too low."""
    configurables = config.get("configurable", {})
    quality_threshold = configurables.get("quality_threshold", 8.0)

    if state["score"] >= quality_threshold:
        # Good enough, proceed to final output
        return state

    llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0.7, max_tokens=8192)
    structured_llm = llm.with_structured_output(RefinementOutput)

    # Format current agents and improvements for refinement
    agents_text = format_agents_for_prompt(state["sub_agents"])
    improvements_text = format_improvements_for_prompt(state["improvements"])

    refinement_prompt = REFINEMENT_PROMPT.format(
        task=state['task'],
        agents_text=agents_text,
        orchestration_strategy=state['orchestration_strategy'],
        score=state['score'],
        critique=state['critique'],
        improvements_text=improvements_text
    )

    response = structured_llm.invoke([HumanMessage(content=refinement_prompt)])

    return {
        **state,
        "sub_agents": response.sub_agents,
        "orchestration_strategy": response.orchestration_strategy,
        "iteration": state.get("iteration", 0) + 1
    }


def finalizer_node(state: AgentOptimizationState) -> AgentOptimizationState:
    """Create the final structured output of the optimized agent system."""

    final_output = {
        "task": state["task"],
        "final_score": state["score"],
        "iterations": state["iteration"],
        "sub_agents": [
            {
                "name": agent.name,
                "description": agent.description,
                "prompt": agent.prompt
            }
            for agent in state["sub_agents"]
        ],
        "orchestration_strategy": state["orchestration_strategy"],
        "critique": state["critique"],
        "improvements_applied": state.get("improvements", [])
    }

    return {
        **state,
        "final_output": final_output
    }


def should_refine(state: AgentOptimizationState, config: RunnableConfig) -> str:
    """Decide whether to refine or finalize based on score and iteration count."""
    configurables = config.get("configurable", {})
    quality_threshold = configurables.get("quality_threshold", 8.0)
    max_refine_iterations = configurables.get("max_refine_iterations", 3)
    if state["score"] >= quality_threshold or state.get("iteration", 0) >= max_refine_iterations:
        return "finalize"
    else:
        return "refine"


# Build the LangGraph workflow
def create_agent_optimizer_graph():
    workflow = StateGraph(AgentOptimizationState)

    # Add nodes
    workflow.add_node("analyze_task", task_analyzer_node)
    workflow.add_node("critique", critic_node)
    workflow.add_node("refine", refinement_node)
    workflow.add_node("finalize", finalizer_node)

    # Add edges
    workflow.set_entry_point("analyze_task")
    workflow.add_edge("analyze_task", "critique")

    # Conditional edge based on score
    workflow.add_conditional_edges(
        "critique",
        should_refine,
        {
            "refine": "refine",
            "finalize": "finalize"
        }
    )

    # After refinement, go back to critique
    workflow.add_edge("refine", "critique")

    # End after finalization
    workflow.add_edge("finalize", END)

    return workflow.compile()
