# Self-Optimizing Deep Agents

Deep Agents have showcased to be a a very powerful concept that is capable of accomplishing many tasks on their own. 
Recent experiments showcase that `Deep Agents` Achieve incredible performance gains with the assistnace of `sub-agents`.
Configuring the right `sub-agents` and ensuring that they have the proper prompts, and even descriptions for other agents has so far been very challenging.
The SODA (Self-Optimizing Deep Agents) project is an Open Source Project that is focused on automating the process of developing 
deep agents for the right use cases, and making discovering / refining the correct sub-agents a breeze.
The intent is to allow users to focus more on the task at hand, and abstract away the `context engineering`

**Acknowledgements** : The following project leverages `LangChain`'s `DeepAgent` framework and is built ontop of it

## Installation
clone this repo and run

```bash
pip install -e .
```

## Usage

Here is how to find the optimal sub-agents and prompt for your deep agent based on the your specific task
(NOTE: Currently, leverages `anthropic claude` so set your `ANTHROPIC_API_KEY` variable)

```python
from soda import optimize_agents_for_task

research_task = "Conduct extensive research for the user. The following task will require searching the web and reasoning about the findings made."

optimal_research_subagents = optimize_agents_for_task(task=research_task)

print(optimal_research_subagents["sub_agents"]) # Optimal Subagents for the task
print(optimal_research_subagents["orchestration_strategy"]) # optimal Deep Agents Main prompt for the task 
```

Once the above is complete create your `DeepAgent` as follows
(NOTE: For the following run `pip install tavily-python` before if `tavily` isn't available.)

```python
import os
from typing import Literal

from tavily import TavilyClient
from deepagents import create_deep_agent

tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

# Search tool to use to do research
def internet_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
):
    """Run a web search"""
    return tavily_client.search(
        query,
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic=topic,
    )

tools = [internet_search]

deep_agent = create_deep_agent(
    tools,
    optimal_research_subagents["orchestration_strategy"], # Optimal prompt
    optimal_research_subagents["sub_agents"] # Optimal sub-agents
)

# Invoke the agent
result = deep_agent.invoke({"messages": [{"role": "user", "content": "what is langChain Deep Agents?"}]})
```

## Roadmap
- [ ] Add a GUI for manging and optimizing Sub-agents 
- [ ] Create a registry system that can keep track of sub-agents and a CLI tool 
- [ ] Add Benchmarks for sub-agent performance and optimization
- [ ] Add feature(s) for life-long learning and refining existing Sub-agents based on task performance

