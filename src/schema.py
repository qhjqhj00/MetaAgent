from pydantic import BaseModel, Field
from typing import List

class reasoning_result(BaseModel):
    status: str = Field(description="The status of the reasoning result")
    content: str = Field(description="The content of the reasoning result")
    rationale: str = Field(description="The rationale of the reasoning result")

class tool_result(BaseModel):
    tool: str = Field(description="The tool to use")
    content: list[str] = Field(description="The content of the tool result")

class reflection_result(BaseModel):
    tool_guide: list[str] = Field(description="The tool guide for the agent to use")
    task_guide: list[str] = Field(description="The task guide for the agent to use")
