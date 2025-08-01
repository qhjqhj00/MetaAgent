from config import Configuration
from pydantic import BaseModel
from openai import OpenAI

def get_reasoning_agent(config: Configuration, use_advanced_reasoning: bool = True) -> tuple[OpenAI, str]:
    if config.advanced_reasoning_model and use_advanced_reasoning:
        llm = OpenAI(
            base_url=config.advanced_reasoning_model_base_url,
            api_key=config.advanced_reasoning_model_api_key,
        )
        llm_name = config.advanced_reasoning_model
    else:
        llm = OpenAI(
            base_url=config.reasoning_model_base_url,
            api_key=config.reasoning_model_api_key,
        )
        llm_name = config.reasoning_model
    return llm, llm_name

def get_auxiliary_agent(config: Configuration) -> OpenAI:
    llm = OpenAI(
        base_url=config.auxiliary_model_base_url,
        api_key=config.auxiliary_model_api_key,
    )
    return llm



