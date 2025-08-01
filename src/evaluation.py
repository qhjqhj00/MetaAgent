from my_own_tools import *
from collections import defaultdict
import re
import string
from openai import OpenAI
import os
def llm_evaluate_equivalence_single(
    client: OpenAI,
    question: str,
    labeled_answer: str,
    pred_answer: str,
    model_name: str,
    retry_limit: int = 3,
) -> bool:
    """Evaluate a single pair of answers using LLM"""

    prompt = f"""You are an evaluation assistant. Please determine if the predicted answer is equivalent to the labeled answer.

Question: {question}

Labeled Answer: {labeled_answer}

Predicted Answer: {pred_answer}

Are these answers equivalent? Please respond with "Correct" if they are equivalent, or "Incorrect" if they are not equivalent. Do not include any other text.
"""


    for attempt in range(retry_limit):
        try:
            chat_response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
            )
            response_text = chat_response.choices[0].message.content.strip()
            llm_judge = response_text.lower() == "correct" and \
                not ("incorrect" in response_text.lower() or \
                        "wrong" in response_text.lower() or \
                        "not correct" in response_text.lower())
            return llm_judge, response_text
        except Exception as e:
            if attempt == retry_limit - 1:
                print(f"Error in LLM evaluation: {e}")
                return False, "Error"
            time.sleep(1 * (attempt + 1))
    
    return False, "Error"

def normalize_answer_qa(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.strip().split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def calculate_metrics_by_level(data):
    # Initialize metrics dictionary for each level
    level_metrics = defaultdict(lambda: {"correct": 0, "total": 0, "f1_scores": [], "llm_equivalence": []})
    
    def tokenize(text):
        # Simple tokenization by splitting on whitespace and removing punctuation
        return [word.lower().strip('.,!?()[]{}') for word in text.split() if word.strip()]
    
    def calculate_f1(pred_tokens, gold_tokens):
        # Calculate precision and recall
        pred_set = set(pred_tokens)
        gold_set = set(gold_tokens)
        
        if not pred_set or not gold_set:
            return 0.0
            
        precision = len(pred_set & gold_set) / len(pred_set)
        recall = len(pred_set & gold_set) / len(gold_set)
        
        if precision + recall == 0:
            return 0.0
            
        return 2 * (precision * recall) / (precision + recall)
    
    # Calculate metrics for each example
    for line in data:
        if "Level" not in line:
            print(line)
            continue
        level = line["Level"]
        level_metrics[level]["total"] += 1
        
        true_answer = normalize_answer_qa(line["answer"])
        if "predicted_answer" not in line:
            print(line)
            continue
        pred_answer = normalize_answer_qa(line["predicted_answer"])
        if true_answer == pred_answer:
            level_metrics[level]["correct"] += 1
        elif pred_answer.find(true_answer) != -1 or true_answer.find(pred_answer) != -1:
            print(line["Question"])
            print("pred_answer: ", pred_answer)
            print("true_answer: ", true_answer)
            print("level: ", level)
            print("-" * 50)
        # Calculate F1
        pred_tokens = tokenize(pred_answer)
        gold_tokens = tokenize(true_answer)
        f1_score = calculate_f1(pred_tokens, gold_tokens)
        level_metrics[level]["f1_scores"].append(f1_score)
        if "llm_equivalence" in line:
            level_metrics[level]["llm_equivalence"].append(line["llm_equivalence"])
        else:
            level_metrics[level]["llm_equivalence"].append(False)
    # Print metrics for each level
    print("\nMetrics by Level:")
    print("-" * 50)
    res = defaultdict(dict)
    for level in sorted(level_metrics.keys()):
        total = level_metrics[level]["total"]
        correct = level_metrics[level]["correct"]
        em = correct / total if total > 0 else 0
        avg_f1 = sum(level_metrics[level]["f1_scores"]) / total if total > 0 else 0
        res[level]["em"] = round(em, 4)
        res[level]["avg_f1"] = round(avg_f1, 4)
        res[level]["llm_equivalence_correct"] = sum(level_metrics[level]["llm_equivalence"]) / total if total > 0 else 0
        res[level]["total"] = total
        res[level]["correct"] = correct
        print(f"Level {level}:")
        print(f"Total samples: {total}")
        print(f"Correct predictions: {correct}")
        print(f"Exact Match (EM): {em:.4f}")
        print(f"LLM Equivalence: {round(res[level]['llm_equivalence_correct'], 4)}")
        print(f"F1 Score: {round(avg_f1, 4)}")
        print("-" * 50)
    
    level_keys = list(res.keys())
    n_levels = len(level_keys)
    
    res["average"]["llm_equivalence_correct"] = round(sum([res[level]["llm_equivalence_correct"]*res[level]["total"] for level in level_keys]) / sum([res[level]["total"] for level in level_keys]), 4)
    res["average"]["total"] = sum([res[level]["total"] for level in level_keys])
    res["average"]["correct"] = sum([res[level]["correct"] for level in level_keys])
    res["average"]["em"] = round(res["average"]["correct"] / res["average"]["total"], 4)
    return res





