import io
import sys
import traceback
from state import OverallState
import json
import re
import random
import numpy as np
from pydantic import BaseModel
import requests
from nltk.tokenize import sent_tokenize
import nltk
import concurrent.futures
import time


def safe_exec(code_str, globals_dict=None, locals_dict=None, timeout=None):
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()

    # 安全的默认执行环境
    if globals_dict is None:
        globals_dict = {"__builtins__": __builtins__}
    if locals_dict is None:
        locals_dict = {}

    try:
        exec(code_str, globals_dict, locals_dict)
        output = sys.stdout.getvalue()
        return {'success': True, 'output': output.strip(), 'error': ''}
    except Exception as e:
        error_msg = traceback.format_exc()
        output = sys.stdout.getvalue()
        return {'success': False, 'output': output.strip(), 'error': error_msg}
    finally:
        sys.stdout = old_stdout

def clean_webpage_content(content: str) -> str:
    """
    Clean webpage content by removing URLs, normalizing placeholders, and filtering out overly long words.
    
    Args:
        content (str): Raw webpage content to clean
        
    Returns:
        str: Cleaned content
    """
    if not content:
        return ""
    
    # Remove URLs (http/https/ftp/www patterns)
    url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+|ftp://[^\s<>"{}|\\^`\[\]]+|www\.[^\s<>"{}|\\^`\[\]]+'
    content = re.sub(url_pattern, '', content)
    
    # Remove email addresses
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    content = re.sub(email_pattern, '', content)
    
    # Replace consecutive placeholder characters with single ones
    # Common placeholders: dots, dashes, underscores, asterisks, equals, newlines
    placeholder_patterns = [
        (r'\.{2,}', '.'),  # Multiple dots to single dot
        (r'-{2,}', '-'),   # Multiple dashes to single dash
        (r'_{2,}', '_'),   # Multiple underscores to single underscore
        (r'\*{2,}', '*'),  # Multiple asterisks to single asterisk
        (r'={2,}', '='),   # Multiple equals to single equal
        (r'\n{2,}', '\n'), # Multiple newlines to single newline
        (r'\s{2,}', ' '),  # Multiple spaces to single space
    ]
    
    for pattern, replacement in placeholder_patterns:
        content = re.sub(pattern, replacement, content)
    
    cleaned_content = content.strip()
    
    return cleaned_content


def process_and_merge_search_results(results, cache_results, task_cache_results, config, max_content_length: int = 30000):
    """
    Process and merge search results from two sources, deduplicate by URL,
    rerank content using rerank service, and format as JSON.
    
    Args:
        results: Search results with keys: title, url, context, snippet
        cache_results: Cache search results with keys: title, url, content, snippet
        
    Returns:
        str: Formatted JSON string with merged and reranked results
    """
    # Combine and deduplicate by URL
    combined_results = {}
    if 'results' in cache_results:
        for result in cache_results['results']:
            url = result.get('url', '')
            if url and url not in combined_results:
                combined_results[url] = {
                    'title': result.get('title', 'No title'),
                    'url': url,
                    'content': clean_webpage_content(result.get('content', ''))[:max_content_length],
                    'snippet': result.get('snippet', '')
                }
    # Process regular search results
    if 'results' in results:
        for result in results['results']:
            url = result.get('url', '')
            if url and url not in combined_results:
                combined_results[url] = {
                    'title': result.get('title', 'No title'),
                    'url': url,
                    'content': clean_webpage_content(result.get('context', ''))[:max_content_length],  # Use context as content
                    'snippet': result.get('snippet', '')
                }
    
    # Process cache search results
    if 'results' in task_cache_results:
        for result in task_cache_results['results']:
            url = result.get('url', '')
            if url and url not in combined_results:
                combined_results[url] = {
                    'title': result.get('title', 'No title'),
                    'url': url,
                    'content': clean_webpage_content(result.get('content', ''))[:max_content_length],
                    'snippet': result.get('snippet', '')
                }
    
    return list(combined_results.values())

def truncate_reasoning_str(reasoning_str: str) -> str:
    reasoning_str = reasoning_str.split("<think>")[-1]
    truncated_reasoning_str = ""
    reasoning_steps = reasoning_str.split("\n\n")
    for i,step in enumerate(reasoning_steps):
        if i == 0 or i >= len(reasoning_steps) - 5 or "<help>" in step or "<evidence>" in step:
            truncated_reasoning_str += f"Step {i+1}: {step}\n\n"
        else:
            if truncated_reasoning_str[-len('\n\n...\n\n'):] != '\n\n...\n\n':
                truncated_reasoning_str += '...\n\n'
    return truncated_reasoning_str.strip('\n')

def format_task_description(state: OverallState) -> str:
    description = f"""
The original task is: {state["messages"][0].content}

The agent's reasoning is: {state["reasoning_str"]}

The agent's final answer is: {state["previous_answer"][-1]}

The true answer is: {state["true_answer"]}

"""
    return description

def set_seed(seed: int = 1234):
    random.seed(seed)
    np.random.seed(seed)

def extract_between(text, start_marker, end_marker):
    """Extracts text between two markers in a string."""
    pattern = re.escape(end_marker [::-1]) + r"(.*?)" + re.escape(start_marker[::-1])
    matches = re.findall(pattern, text[::-1], flags=re.DOTALL)
    if matches:
        return matches[0][::-1].strip()
    return None

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_color(text, color):

    print(f"{color}{text}{bcolors.ENDC}")

def stream_completion(agent, model_name, prompt, stop=None, note=None, stream=True, schema: BaseModel = None, max_tokens: int = 10000, top_p: float = 0.8, temperature: float = 0.7, repetition_penalty: float = 1.05, min_p: float = 0.05, top_k: int = 20):
    
    num_try = 0
    while num_try < 5:
        try:
            response = agent.completions.create(
                model=model_name,
                prompt=prompt,
                max_tokens=max_tokens,
                top_p=top_p,
                temperature=temperature,
                stream=stream,
                stop=stop,
                extra_body={
                    "min_p": min_p,
                    "repetition_penalty": repetition_penalty,
                    'include_stop_str_in_output': True,
                    'top_k': top_k,
                    "guided_json": schema.model_json_schema() if schema else None
                }
            )
            break
        except Exception as e:
            print(f"Error: {e}")
            num_try += 1
            time.sleep(1)

    if note:
        print_color(note, bcolors.OKGREEN)
    if stream:
        response_content = ""   
        for chunk in response:
            response_content += chunk.choices[0].text
            print(chunk.choices[0].text, end="", flush=True)
        return response_content
    else:
        return response.choices[0].text


def batch_completion(agent, model_name, prompts: list, max_tokens: int = 10000, top_p: float = 0.8, temperature: float = 0.7, repetition_penalty: float = 1.05, min_p: float = 0.05, top_k: int = 20) -> list:
    """Process multiple prompts in parallel using ThreadPoolExecutor"""
    print(f"Processing {len(prompts)} prompts in parallel...")
    results = [None] * len(prompts)  # Initialize a list with the same length as prompts
    
    # Define a worker function for threading
    def worker(index, prompt):
        result = stream_completion(
            agent=agent,
            model_name=model_name,
            prompt=prompt,
            max_tokens=max_tokens,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            min_p=min_p,
            top_k=top_k,
            stream=False  # Disable streaming for batch processing
        )
        results[index] = result
    
    # Use ThreadPoolExecutor for concurrent requests
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Map each prompt to the worker function
        futures = {executor.submit(worker, idx, prompt): idx for idx, prompt in enumerate(prompts)}
        
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()  # Get task result, exceptions will be caught if thrown
            except Exception as exc:
                print(f'Generated an exception: {exc}')
                idx = futures[future]
                results[idx] = None
    
    return results

if __name__ == "__main__":
    text = "# Example ISBN-like numbers (replace with actual data if available)\nisbn_numbers = [\n    '9780306406157',\n    '9780306406517',\n    '9780306406127'\n]\n\n# Search for possible (weight, transposed_column) pairs\ndef check_isbn_pair(isbn_list):\n    valid_pairs = []\n    for weight in range(1, 10):\n        for col in range(3, 11):\n            all_valid = True\n            for isbn in isbn_list:\n                digits = [int(x) for x in isbn]\n                # Transpose columns col and col+1\n                if col+1 >= len(digits):\n                    all_valid = False\n                    break\n                digits[col], digits[col+1] = digits[col+1], digits[col]\n                # Calculate checksum with the unknown weight at col\n                checksum = sum(d * (i+1 if i != col else weight) for i, d in enumerate(digits[:-1]))\n                if checksum % 11 != digits[-1]:\n                    all_valid = False\n                    break\n            if all_valid:\n                valid_pairs.append((weight, col))\n    print(valid_pairs)\n\ncheck_isbn_pair(isbn_numbers)"
    print(safe_exec(text))