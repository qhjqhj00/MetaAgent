from langgraph.graph import add_messages
from typing_extensions import Annotated
import operator
from typing import TypedDict, Optional

class OverallState(TypedDict):
    messages: Annotated[list, add_messages]
    help_content: Annotated[list, operator.add]
    tool_selection: Annotated[list, operator.add]
    tool_content: Annotated[list, operator.add]
    tool_result: Annotated[list, operator.add]
    status: Annotated[list, operator.add]
    previous_critical_thinking: Annotated[list, operator.add]
    previous_answer: Annotated[list, operator.add]
    reasoning_str: str
    experience: str
    try_count: int = 0
    true_answer: Optional[str] = None
    llm_equivalence: Optional[bool] = None
    
