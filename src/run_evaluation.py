from graph import get_graph
from langchain_core.messages import HumanMessage
from config import Configuration
from datetime import datetime
from evaluation import calculate_metrics_by_level
from collections import defaultdict
import json

def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

config = Configuration.from_runnable_config()
graph = get_graph(config)

print(config, "\n", "="*100, "\n")

task = config.eval_task
if task == "GAIA":
    data = load_json("data/GAIA/dev.json")
elif task == "webwalker":
    data = load_json("data/webwalker/test.json")
elif task == "BrowseComp":
    data = load_json("data/BrowseComp/subset.json")
else:
    raise ValueError(f"Invalid task: {task}")

save_file = f"data/{task}/{config.version}.{datetime.now().strftime('%Y-%m-%d_%H:%M')}.jsonl"
with open(save_file, "w") as f:
    json.dump(config.__dict__, f, ensure_ascii=False)
    f.write("\n")

experience_list = []
for i,line in enumerate(data):
    question = line["Question"]
    answer = line["answer"]

    success = False
    for attempt in range(config.max_retries):
        try:
            result = graph.invoke({"messages": [{"role": "user", "content": question}], "evidence": [], "true_answer": answer, "reasoning_str": "", "experience": experience_list[-1:]},{"recursion_limit": 50})
            line["predicted_answer"] = result["messages"][-1].content
            line["tool_selection"] = result["tool_selection"]
            line["tool_content"] = result["tool_content"]
            line["tool_result"] = result["tool_result"]
            line["previous_critical_thinking"] = result["previous_critical_thinking"]
            line["previous_answer"] = result["previous_answer"]
            line["reasoning_str"] = result["reasoning_str"]
            line["reasoning_status"] = result["status"]
            line["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            line["status"] = "success"
            line["attempts"] = attempt + 1
            line["experience"] = result["experience"]
            
            line["llm_equivalence"] = result["llm_equivalence"]
            if result["experience"] != "":
                experience_list.append([question, result["experience"]])
            success = True
            break
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt == config.max_retries - 1:
                line["status"] = "error"
                line["error"] = str(e)
                line["attempts"] = config.max_retries
            else:
                continue
    if "predicted_answer" not in line:
        line["predicted_answer"] = ""
    if config.eval_task == "webwalker":
        line["Level"] = line["difficulty_level"]
    elif config.eval_task in ["BrowseComp"]:
        line["Level"] = line["problem_topic"]
    with open(save_file, "a") as f:
        json.dump(line, f, ensure_ascii=False)
        f.write("\n")
    print("="*100)

level_metrics = calculate_metrics_by_level(data)
level_metrics["save_file"] = save_file
with open(f"data/{task}.metrics.jsonl", "a") as f:
    f.write(json.dumps(level_metrics, ensure_ascii=False))
    f.write("\n")




