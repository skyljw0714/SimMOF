from pydantic import BaseModel
from typing import List


class WorkflowStep(BaseModel):
    step: int
    tool: str
    condition: str
    reason: str


class Workflow(BaseModel):
    goal: str
    steps: List[WorkflowStep]
