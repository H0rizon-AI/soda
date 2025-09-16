from langchain_core.messages import SystemMessage, HumanMessage
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, END

from src.soda.types import AgentOptimizationState, CritiqueOutput,RefinementOutput,TaskAnalysisOutput


def task_analyzer_node(state: AgentOptimizationState) -> AgentOptimizationState:
    """Analyze the task and generate initial sub-agents and orchestration strategy."""

    llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0.7, max_tokens=8192)
    structured_llm = llm.with_structured_output(TaskAnalysisOutput)

    analysis_prompt = f"""
    Given this task: "{state['task']}"

    Analyze the task and generate:
    1. A list of sub-agents needed to complete this task
    2. An orchestration strategy that explains how these agents should work together

    For each sub-agent, provide:
    - name: A clear, specific name for the sub-agent
    - description: What this agent does (visible to other agents)
    - prompt: The internal system prompt for this specific agent

    Think about:
    - What are the distinct capabilities needed?
    - How should work be divided?
    - What are the dependencies between agents?
    - What's the optimal workflow?
    - How can the file system be leveraged? 
    - What is the final return of this Particular Agent when its finished?

    Make sure each sub-agent has a clear, specific role and the orchestration strategy
    explains the complete workflow from start to finish.
    """

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
    agents_text = "\n".join([
        f"Agent: {agent.name}\n"
        f"Description: {agent.description}\n"
        f"Prompt: {agent.prompt}\n"
        for agent in state["sub_agents"]
    ])

    critique_prompt = f"""
    Original Task: "{state['task']}"

    Generated Sub-Agents:
    {agents_text}

    Orchestration Strategy:
    {state['orchestration_strategy']}

    Please critique this agent design on the following dimensions:

    1. **Completeness**: Do these agents cover all aspects of the task?
    2. **Clarity**: Are the agent roles and prompts clear and specific?
    3. **Efficiency**: Is the work divided optimally? Any redundancy or gaps?
    4. **Coordination**: Will these agents work well together? Clear handoffs?
    5. **Feasibility**: Are the individual agent responsibilities realistic?

    Provide:
    - A score from 1-10 (10 being perfect)
    - Detailed analysis of strengths and weaknesses
    - Concrete suggestions for better agent design

    Consider the complexity of the original task and whether this agent design
    appropriately handles that complexity.
    """

    response = structured_llm.invoke([HumanMessage(content=critique_prompt)])

    return {
        **state,
        "critique": response.critique,
        "score": response.score,
        "improvements": response.improvements
    }


def refinement_node(state: AgentOptimizationState) -> AgentOptimizationState:
    """Refine the agents and strategy based on critique if score is too low."""

    if state["score"] >= QUALITY_THRESHOLD:
        # Good enough, proceed to final output
        return state

    llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0.7, max_tokens=8192)
    structured_llm = llm.with_structured_output(RefinementOutput)

    # Format current agents for refinement
    agents_text = "\n".join([
        f"Agent: {agent.name}\n"
        f"Description: {agent.description}\n"
        f"Prompt: {agent.prompt}\n"
        for agent in state["sub_agents"]
    ])

    improvements_text = "\n".join([f"- {improvement}" for improvement in state["improvements"]])

    refinement_prompt = f"""
    Original Task: "{state['task']}"

    Current Sub-Agents:
    {agents_text}

    Current Orchestration Strategy:
    {state['orchestration_strategy']}

    Critique (Score: {state['score']}/10):
    {state['critique']}

    Specific Improvements Needed:
    {improvements_text}

    Based on the critique and specific improvement suggestions, improve the agent design. 
    Address the specific issues raised and create a better agent architecture.

    Focus on:
    - Fixing identified gaps or redundancies
    - Improving role clarity and coordination
    - Enhancing the orchestration strategy
    - Making agent prompts more specific and actionable
    """

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


def should_refine(state: AgentOptimizationState) -> str:
    """Decide whether to refine or finalize based on score and iteration count."""
    if state["score"] >= QUALITY_THRESHOLD or state.get("iteration", 0) >= MAX_ITERATIONS:
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


def optimize_agents_for_task(task: str, max_iterations: int = MAX_ITERATIONS,
                             quality_threshold: float = QUALITY_THRESHOLD):
    """Run the full optimization process for a given task."""

    # Update global config if different values provided
    global MAX_ITERATIONS, QUALITY_THRESHOLD
    MAX_ITERATIONS = max_iterations
    QUALITY_THRESHOLD = quality_threshold

    graph = create_agent_optimizer_graph()

    initial_state = AgentOptimizationState(
        task=task,
        sub_agents=[],
        orchestration_strategy="",
        critique="",
        score=0.0,
        improvements=[],
        iteration=0,
        final_output={}
    )

    result = graph.invoke(initial_state)
    return result["final_output"]