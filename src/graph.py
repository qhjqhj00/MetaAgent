from typing import Dict, List, TypedDict, Annotated
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, AIMessage
from config import Configuration
from agent import get_reasoning_agent, get_auxiliary_agent
from langchain_core.runnables import RunnableConfig
from langgraph.graph import add_messages
from prompts import *
from state import *
from schema import reasoning_result, tool_result, reflection_result
import json 
from utils import *
import requests
from itertools import chain
import re
from evaluation import llm_evaluate_equivalence_single

def cache_search(query: str, config: Configuration) -> str:
    cache_search_url = config.cache_search_url
    cache_search_data = {
        "query": query,
        "topk": config.cache_search_topk
    }
    cache_search_response = requests.post(cache_search_url, json=cache_search_data)
    cache_search_response.raise_for_status()
    cache_search_result = cache_search_response.json()
    return cache_search_result

def web_search(queries: list[str], reasoning_str: str, oringinal_task: str, search_intention: str, config: Configuration, first_search: bool) -> str:
    if config.use_web_search:
        results = {"results": []}
        for q in queries[:3]: # limit the number of queries for web search to 3
            search_url = f"{config.search_api_url}?query={q}&topk={config.search_topk}"
            response = requests.get(search_url)
            results["results"].extend(response.json()["results"])
    else:
        results = {"results": []}
    print("web search results: ", len(results["results"]))

    if config.use_cache_search:
        cache_search_result = {"results": []}
        for q in queries:
            cache_search_result["results"].extend(cache_search(q, config)["results"])
    else:
        cache_search_result = {"results": []}
    print("cache search results: ", len(cache_search_result["results"]))
    if first_search:
        task_cache_search = cache_search(oringinal_task, config)
    else:
        task_cache_search = {}

    if not results and not cache_search_result:
        raise ValueError("No search results found")

    
    web_search_result = process_and_merge_search_results(results, cache_search_result, task_cache_search, config, max_content_length=30000)
    prev_reasoning = truncate_reasoning_str(reasoning_str)
    prompts = []

    for i, result in enumerate(web_search_result):
        refine_search_result_prompt = get_webpage_to_reasonchain_instruction(prev_reasoning, search_intention, oringinal_task, result["content"])

        refine_search_result_prompt = qwen_think_template.format(prompt=refine_search_result_prompt)
        prompts.append(refine_search_result_prompt)

    agent, agent_name = get_reasoning_agent(config, use_advanced_reasoning=False)
    content = batch_completion(
        agent, 
        agent_name, 
        prompts, 
        max_tokens=20000,
    )
    evidence = []

    for refined_content in content:
        if "</think>" not in refined_content:
            res = refined_content
            print_color(f"No think in refined_content: {refined_content}", bcolors.WARNING)
        else:
            res = refined_content.split("</think>")[1]
        if res.find("No helpful information found") != -1:
            continue
        res = extract_between(res, "<evidence>", "</evidence>")
        if res is not None:
            evidence.append(res)

    evidence_str = ""
    if len(evidence) == 0:
        evidence_str = "No helpful information found, you should try different angle to ask for help or try to answer the question by yourself."
    for i, c in enumerate(evidence):
        evidence_str += f"Web page {i+1}: {c}\n\n"
    print_color(evidence_str, bcolors.OKGREEN)
    
    return web_search_result, evidence_str

def calculator(state: OverallState, config: RunnableConfig) -> OverallState:
    to_execute = state["tool_content"][-1]
    if isinstance(to_execute, list):
        to_execute = to_execute[0]
    try:
        res = eval(to_execute)
    except:
        res = safe_exec(to_execute)
        if res['success']:
            res = res['output']
        else:
            res = "Error: " + res['error']
    print_color("calculator result: ", bcolors.OKGREEN)
    print(res)
    state["reasoning_str"] += f"Code execution tool provided the following result: {res}"
    return {
        "tool_result": [res]
    }

def search_node(state: OverallState, config: RunnableConfig) -> OverallState:
    configurable = Configuration.from_runnable_config(config)
    queries = state["tool_content"][-1]
    searched_query = chain(*state["tool_content"][:-1])
    search_intention = state["help_content"][-1]
    if len(state["tool_content"]) == 1:
        first_search = True
    else:
        first_search = False
    
    query_to_search = []
    for q in queries:
        if q not in query_to_search:
            query_to_search.append(q)

    if len(query_to_search) == 0:
        print_color("query already searched", bcolors.OKBLUE)
        result_index = state["help_content"].index(search_intention)
        query_result = state["tool_result"][result_index]
        state["reasoning_str"] += f"You have already searched the internet for the intention: {search_intention}. Search tool provided the following result: {query_result}"
        return {
            "reasoning_str": state["reasoning_str"]
        }
    results, evidence = web_search(query_to_search, state["reasoning_str"], state["messages"][0].content, search_intention, configurable, first_search)
    
    state["reasoning_str"] += f"\n\n<evidence> Search tool provided the following results: {evidence} </evidence>\n\n"
    return {
        "tool_result": [evidence],
        "reasoning_str": state["reasoning_str"]
    }


def router_node(state: OverallState, config: RunnableConfig) -> OverallState:
    configurable = Configuration.from_runnable_config(config)

    if state["status"][-1] == "correct_answer":
        return "summarize_experience" if state["true_answer"] is not None and configurable.use_experience else END

    if state["status"][-1] == "answer":
        return "reflection" 

    elif state["status"][-1] == "help":
        help_content = state["help_content"][-1]
        tool_prompt = get_tool_prompt(help_content, state["tool_selection"], state["tool_content"], state["tool_result"])

        tool_prompt = qwen_no_think_template.format(prompt=tool_prompt)
        auxiliary_agent = get_auxiliary_agent(configurable)
        auxiliary_agent_name = configurable.auxiliary_model

        response_content = stream_completion(
            auxiliary_agent, 
            auxiliary_agent_name, 
            tool_prompt, 
            note="tool response: ",
            stream=False,
            schema=tool_result
        )

        response = json.loads(response_content)
        print_color("tool response: ", bcolors.HEADER)
        print(response)

        state["tool_selection"].append(response["tool"])
        state["tool_content"].append(response["content"])
        if response["tool"] == "search":
            return "search"
        elif response["tool"] == "calculator":
            return "calculator" 
        elif response["tool"] == "code_execution":
            return "calculator"
        else:
            raise ValueError("Invalid tool")
    else:
        raise ValueError("Invalid status")

def summarize_experience_node(state: OverallState, config: RunnableConfig) -> OverallState:
    
    configurable = Configuration.from_runnable_config(config)
    task_description = format_task_description(state)

    prompt = get_summarize_experience_prompt(task_description, state["experience"])
    prompt = qwen_think_template.format(prompt=prompt)

    agent, agent_name = get_reasoning_agent(configurable)

    analysis = stream_completion(
        agent, 
        agent_name, 
        prompt, 
        note="summarize experience: "
    )

    experience = extract_between(analysis, "<updated_experience>", "</updated_experience>")
    return {
        "experience": experience
    }

def reflection_node(state: OverallState, config: RunnableConfig) -> OverallState:
    configurable = Configuration.from_runnable_config(config)
    if len(state["previous_critical_thinking"]) > 3:
        return {
            "messages": AIMessage(content=random.choice(state["previous_answer"])),
            "status": "correct_answer"
        }
    
    question = state["messages"][0].content
    reasoning = state["reasoning_str"]
    answer = state["previous_answer"][-1]
    if configurable.use_reflection:
        prompt = get_reflection_prompt(question, reasoning, answer)
        prompt = qwen_think_template.format(prompt=prompt)
        reasoning_agent, reasoning_agent_name = get_reasoning_agent(configurable)
        response = stream_completion(
            reasoning_agent, 
            reasoning_agent_name, 
            prompt, 
            note="critical thinking response: "
        )
        analysis = extract_between(response, "<analysis>", "</analysis>")
        analysis = analysis.split("\n")
        answer_correctness = analysis[0].split(": ")[1].strip()
        suggested_answer = analysis[1].split(": ")[1].strip()
    else:
        answer_correctness = "correct"
        suggested_answer = "n/a"
        response = ""

    if answer_correctness == "correct":
        
        auxiliary_agent = get_auxiliary_agent(configurable)
        auxiliary_agent_name = configurable.auxiliary_model
        llm_equivalence, response_text = llm_evaluate_equivalence_single(
            auxiliary_agent, 
            question, 
            state["true_answer"], 
            answer, 
            auxiliary_agent_name
        )
        print_color(f"llm_equivalence: {llm_equivalence}, response_text: {response_text}", bcolors.OKGREEN)
        if suggested_answer != "n/a":
            answer = suggested_answer

        return {
            "messages": AIMessage(content=answer), 
            "status": ["correct_answer"],
            "previous_critical_thinking": [response],
            "llm_equivalence": llm_equivalence
        }

    elif answer_correctness == "incorrect":
        return {
            "status": ["incorrect_answer"],
            "previous_critical_thinking": [response]
        }
    else:
        return { "previous_critical_thinking": [response] }

def reasoning_node(state: OverallState, config: RunnableConfig) -> OverallState:
    print(state["status"])
    print_color(state["experience"], bcolors.OKGREEN)
    configurable = Configuration.from_runnable_config(config)
    if len(state["status"]) > 1 and state["status"][-1] == "correct_answer":
        return {}

    if not state["reasoning_str"] or state["status"][-1] == "incorrect_answer":
        prompt = get_qa_prompt_reasoning(state["messages"][0].content, state["experience"], state["previous_critical_thinking"])
        prompt = qwen_think_template.format(prompt=prompt)
        state["reasoning_str"] = prompt
    elif len(state["status"]) > 15 or len(state["reasoning_str"]) > 120000:
        prompt = state["reasoning_str"] +"\n\n You have reached the maximum number of reasoning steps. Please provide your final answer.\n\n"
    else:
        prompt = state["reasoning_str"]
        
    agent, agent_name = get_reasoning_agent(configurable)

    if agent_name in ["QwQ-32B", "Qwen3-32B"]:
        stop=["</help>", "</answer>"]
    else:
        stop=["<|im_end|>","</help>", "</answer>"]
    response_content = stream_completion(
        agent, 
        agent_name, 
        prompt,
        max_tokens=30000,
        stop=stop, 
        note="reasoning response: ")

    if "<answer>" in response_content and not response_content.endswith("</answer>"):
        response_content += "</answer>"
    elif "<help>" in response_content and not response_content.endswith("</help>"):
        response_content += "</help>"


    if response_content.endswith("</answer>"):
        answer = extract_between(response_content, "<answer>", "</answer>")
        state["reasoning_str"] += response_content
        return {
            "previous_answer": [answer], 
            "status": ["answer"],
            "reasoning_str": state["reasoning_str"]
        }

    elif response_content.endswith("</help>"):
        help_content = extract_between(response_content, "<help>", "</help>")
        response_content_remove_mark = response_content.replace("</think>", "")

        state["reasoning_str"] += response_content_remove_mark

        return {
            "help_content": [help_content],
            "status": ["help"],
            "reasoning_str": state["reasoning_str"]
        }
    else:
        raise ValueError("Invalid response")

def get_graph(config: Configuration):  

    workflow = StateGraph(OverallState, config_schema=config)

    workflow.add_node("router", router_node)
    workflow.add_node("reasoning", reasoning_node)
    workflow.add_node("search", search_node)
    workflow.add_node("calculator", calculator)
    workflow.add_node("reflection", reflection_node)
    workflow.add_node("summarize_experience", summarize_experience_node)

    workflow.add_edge(START, "reasoning")
    workflow.add_conditional_edges("reasoning", router_node, ["search", "calculator",  "reflection", "summarize_experience", END])
    workflow.add_edge("search", "reasoning")
    workflow.add_edge("calculator", "reasoning")
    workflow.add_edge("reflection", "reasoning")
    workflow.add_edge("summarize_experience", END)
    app = workflow.compile()
    return app

