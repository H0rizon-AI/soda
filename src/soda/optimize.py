from soda.types import AgentOptimizationState
from soda.workflow import create_agent_optimizer_graph

def optimize_agents_for_task(task: str, max_refine_iterations: int = 3,
                             quality_threshold: float = 8.0):
    """Run the full optimization process for a given task."""

    # Use config defaults if no values provided
    max_iter = max_refine_iterations
    quality_thresh = quality_threshold
    runtime_config = {
        "recursion_limit": 1000,
        "configurable": {  # Your custom config fields
            "quality_score": quality_thresh,
            "max_refine_iterations" : max_iter
        }
    }

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

    result = graph.invoke(initial_state, config=runtime_config)
    return result["final_output"]