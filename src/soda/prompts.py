TASK_ANALYSIS_PROMPT = """
Given this task: "{task}"

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

CRITIQUE_PROMPT = """
Original Task: "{task}"

Generated Sub-Agents:
{agents_text}

Orchestration Strategy:
{orchestration_strategy}

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

REFINEMENT_PROMPT = """
Original Task: "{task}"

Current Sub-Agents:
{agents_text}

Current Orchestration Strategy:
{orchestration_strategy}

Critique (Score: {score}/10):
{critique}

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


def format_agents_for_prompt(agents):
    """Format sub-agents list for inclusion in prompts."""
    return "\n".join([
        f"Agent: {agent.name}\n"
        f"Description: {agent.description}\n"
        f"Prompt: {agent.prompt}\n"
        for agent in agents
    ])


def format_improvements_for_prompt(improvements):
    """Format improvements list for inclusion in prompts."""
    return "\n".join([f"- {improvement}" for improvement in improvements])