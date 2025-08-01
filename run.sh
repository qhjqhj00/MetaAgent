

python src/run_evaluation.py \
    --reasoning_model QwQ-32B \
    --reasoning_model_base_url http://localhost:12345/v1/ \
    --reasoning_model_api_key empty \
    --auxiliary_model Qwen2.5-7B-Instruct \
    --auxiliary_model_base_url http://localhost:12346/v1/ \
    --auxiliary_model_api_key empty \
    --search_api_url http://localhost:12347/search \
    --cache_search_url http://localhost:12348/search \
    --max_retries 3 \
    --search_topk 10 \
    --use_experience \
    --use_web_search \
    --use_cache_search \
    --use_llm_equivalence \
    --eval_task GAIA \
    --version v1 \
    # --advanced_reasoning_model google/gemini-2.5-flash \
    # --advanced_reasoning_model_base_url https://openrouter.ai/api/v1 \
    # --advanced_reasoning_model_api_key openrouter-api-key
