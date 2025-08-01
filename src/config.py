import os
from pydantic import BaseModel, Field
from typing import Any, Optional

from langchain_core.runnables import RunnableConfig
import argparse

class ConfigParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("--reasoning_model", type=str, default="")
        self.parser.add_argument("--reasoning_model_api_key", type=str, default="")
        self.parser.add_argument("--reasoning_model_base_url", type=str, default="")
        self.parser.add_argument("--auxiliary_model", type=str, default="")
        self.parser.add_argument("--auxiliary_model_base_url", type=str, default="")
        self.parser.add_argument("--auxiliary_model_api_key", type=str, default="")
        self.parser.add_argument("--advanced_reasoning_model", type=str, default="")
        self.parser.add_argument("--advanced_reasoning_model_base_url", type=str, default="")
        self.parser.add_argument("--advanced_reasoning_model_api_key", type=str, default="")
        self.parser.add_argument("--use_web_search", action='store_true', default=False)
        self.parser.add_argument("--use_cache_search", action='store_true', default=False)
        self.parser.add_argument("--search_api_url", type=str, default="")
        self.parser.add_argument("--cache_search_url", type=str, default="")
        self.parser.add_argument("--eval_task", type=str, default="GAIA")
        self.parser.add_argument("--max_retries", type=int, default=3)
        self.parser.add_argument("--version", type=str, default="v1")
        self.parser.add_argument("--search_topk", type=int, default=10)
        self.parser.add_argument("--use_llm_equivalence", action='store_true', default=False)
        self.parser.add_argument("--use_experience", action='store_true', default=False)
        self.parser.add_argument("--use_reflection", action='store_true', default=False)
        self.args = self.parser.parse_args()



class Configuration(BaseModel):
    """The configuration for the agent."""

    reasoning_model: str = Field(
        default="",
        metadata={
            "description": "The name of the language model to use for the agent's reasoning."
        },
    )
    reasoning_model_api_key: str = Field(
        default="",
        metadata={
            "description": "The API key of the language model to use for the agent's reasoning."
        },
    )
    reasoning_model_base_url: str = Field(
        default="",
        metadata={
            "description": "The base URL of the language model to use for the agent's reasoning."
        },
    )
    auxiliary_model_api_key: str = Field(
        default="",
        metadata={
            "description": "The API key of the language model to use for the agent's auxiliary."
        },
    )
    auxiliary_model: str = Field(
        default="",
        metadata={
            "description": "The name of the language model to use for the agent's auxiliary."
        },
    )
    auxiliary_model_base_url: str = Field(
        default="",
        metadata={
            "description": "The base URL of the language model to use for the agent's auxiliary."
        },
    )
    advanced_reasoning_model: Optional[str] = Field(
        default=None,
        metadata={
            "description": "The name of the language model to use for the agent's advanced reasoning."
        },
    )
    advanced_reasoning_model_base_url: Optional[str] = Field(
        default=None,
        metadata={
            "description": "The base URL of the language model to use for the agent's advanced reasoning."
        },
    )
    advanced_reasoning_model_api_key: Optional[str] = Field(
        default=None,
        metadata={
            "description": "The API key of the language model to use for the agent's advanced reasoning."
        },
    )
    max_retries: int = Field(
        default=3,
        metadata={"description": "The maximum number of iterations to perform."},
    )
    search_api_url: str = Field(
        default="",
        metadata={"description": "The URL of the search API."},
    )
    search_topk: int = Field(
        default=10,
        metadata={"description": "The number of search results to return."},
    )
    use_web_search: bool = Field(
        default=False,
        metadata={"description": "Whether to use the web search."},
    )
    cache_search_url: str = Field(
        default="",
        metadata={"description": "The URL of the cache search API."},
    )
    cache_search_topk: int = Field(
        default=10,
        metadata={"description": "The number of cache search results to return."},
    )
    use_cache_search: bool = Field(
        default=True,
        metadata={"description": "Whether to use the cache search."},
    )
    use_llm_equivalence: bool = Field(
        default=True,
        metadata={"description": "Whether to use the LLM equivalence."},
    )
    use_experience: bool = Field(
        default=True,
        metadata={"description": "Whether to use the experience."},
    )
    use_reflection: bool = Field(
        default=True,
        metadata={"description": "Whether to use the reflection."},
    )
    eval_task: str = Field(
        default="GAIA",
        metadata={"description": "The task to evaluate."},
    )
    version: str = Field(
        default="v1",
        metadata={"description": "The version of the evaluation."},
    )
    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )
        parser = ConfigParser()

        # Get raw values from environment or config
        raw_values: dict[str, Any] = {
            name: os.environ.get(name.upper(), configurable.get(name))
            for name in cls.model_fields.keys()
        }

        # Filter out None values
        values = {k: v for k, v in raw_values.items() if v is not None}
        for k, v in parser.args.__dict__.items():
            values[k] = v

        return cls(**values)
