from typing import List, Dict, Any, TypedDict
from pydantic import BaseModel, Field


class SubAgent(BaseModel):
    name: str = Field(description="Name of the sub-agent itself")
    description: str = Field(description="The description to other agents about what does this agent do")
    prompt: str = Field(description="The internal prompt to the sub-agent itself, what the agent will see as its system prompt")

class TaskAnalysisOutput(BaseModel):
    sub_agents: List[SubAgent] = Field(description="List of generated sub-agents needed for the task")
    orchestration_strategy: str = Field(description="Detailed explanation of how agents work together, workflow, handoffs, etc.")

class CritiqueOutput(BaseModel):
    score: float = Field(description="Quality score from 1-10 (10 being perfect)", ge=1.0, le=10.0)
    critique: str = Field(description="Detailed analysis of strengths and weaknesses")
    improvements: List[str] = Field(description="Specific suggestions for better agent design")

class RefinementOutput(BaseModel):
    sub_agents: List[SubAgent] = Field(description="Improved list of sub-agents")
    orchestration_strategy: str = Field(description="Improved explanation of how agents work together")

class AgentOptimizationState(TypedDict):
    task: str  # The original user task
    sub_agents: List[SubAgent]  # Generated sub-agents
    orchestration_strategy: str  # How agents work together
    critique: str  # Reflection and scoring feedback
    score: float  # Quality score from critique
    improvements: List[str]  # Specific improvement suggestions
    iteration: int  # Current iteration number
    final_output: Dict[str, Any]  # Final structured output