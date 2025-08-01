
# <div align="center">MetaAgent: Toward Self-Evolving Agent via Tool Meta-Learning<div>

<div align="center">
<p><strong>Learning by Doing: From Novice to Expert!</strong></p>
<a href="https://arxiv.org/" target="_blank"><img src=https://img.shields.io/badge/arXiv-b5212f.svg?logo=arxiv></a>
<a href="https://github.com/"><img alt="License" src="https://img.shields.io/badge/Apache-2.0-green"></a>
<a href="https://www.langchain.com/langgraph"><img alt="License" src="https://img.shields.io/badge/powered by-LangGraph-blue"></a>
</div>

<h4 align="center">
<p>
<a href="#rocket-infrastructure">Infrastructure</a> |
<a href="#notebook-quick-start">Quick-Start</a> |
<a href="#raised_hands-faqs"> FAQs</a> 
</p>

## Overview
MetaAgent is a next-generation agentic AI framework built on the principle of learning-by-doing: expertise is developed through hands-on practice and continual self-improvement, not through static rules or costly retraining. MetaAgent begins with a minimal workflow, equipped only with essential reasoning and adaptive help-seeking capabilities. When encountering knowledge gaps, it generates natural language help requests, which are flexibly routed to external tools by a dedicated tool router.

As MetaAgent solves tasks, it performs ongoing self-reflection and answer verification, turning experience into concise, generalizable lessons that are dynamically integrated into future tasks. Over time, MetaAgent autonomously builds in-house tools and a persistent knowledge base by organizing its tool-use history—enabling ever more efficient information retrieval and integration.
We call this continual, data-driven improvement process meta tool learning. Unlike traditional agentic systems, MetaAgent evolves and adapts on-the-fly—without changing model weights or needing additional training.

<p align="center">
<img src="figure/model.png" width="1000">
</p>

## :sparkles: Features
- **Minimal, General Workflow:** Starts with core reasoning and help-seeking abilities, adaptable to diverse tasks.  
- **Flexible Tool Routing:** Dedicated router maps natural language requests to appropriate external tools.  
- **Meta Tool Learning:** Uses self-reflection and answer verification to improve reasoning and tool use.  
- **Dynamic Context Engineering:** Distills experience into lessons that update future inputs without retraining.  
- **Persistent Knowledge Base:** Builds and updates internal memory from tool interactions for better retrieval.

## :rocket: Infrastructure

Efficient infrastructure is crucial for deep search applications, as it helps save resources and improve overall performance. Here, we provide a service that combines Google Search and Jina for web page retrieval. This service exposes an HTTP API that accepts two parameters: `query` and `topk`. It first uses the Google Search API to obtain search results, then leverages the Jina Reader API to fetch the content of the resulting web pages.

To optimize resource usage, we deploy a MySQL service to cache both search results and web page contents. This ensures that repeated queries are served quickly without redundant external requests.

### Setup Instructions

1. **Deploy MySQL with Docker**

   If this is your first time using the service, pull the MySQL Docker image:

   ```bash
   docker pull mysql:8.0
   ```

2. **Start MySQL Container with Custom User, Password, and Port**

   Run the MySQL container with your own username, password, and port (e.g., 3306):

   ```bash
   docker run -d \
     --name metaagent-mysql \
     -e MYSQL_ROOT_PASSWORD=your_password \
     -e MYSQL_USER=your_user \
     -e MYSQL_PASSWORD=your_password \
     -e MYSQL_DATABASE=search \
     -p 3306:3306 \
     mysql:8.0
   ```

   > Replace `your_user` and `your_password` with your desired username and password. You can also change `-p 3306:3306` to use a different port if needed.

   After starting, the MySQL service will be available on your local machine at port 3306.

   Check the script at `tools/db/build_db.py` and use the `build_db(db_name)` function to create the necessary database structure.

3. **Configure API Keys**

   Update the configuration file at `data/api_dict.json` with your Google Search API key, CSE ID, and Jina API key. Example:

   ```json
   {
       "db": {
           "root": {
               "host": "localhost",
               "port": 3306,
               "user": "your_user",
               "password": "your_password"
           }
       },
       "jina": {
           "api_key": "your_jina_api_key"
       },
       "search_engine": {
           "google": {
               "api_key": "your_google_api_key",
               "cse_id": "your_cse_id"
           }
       }
   }
   ```

4. **Run the Search Service**

   Start the search and web scraping service locally:

   ```bash
   python -m tools.db.search_app
   ```

   You can configure the port and other settings in the `tools/db/search_app.py` file. If needed, you can also replace this service with your own custom search backend.

This setup provides a robust, resource-efficient search infrastructure for deep search tasks, combining fast retrieval with persistent caching.

A case:

```bash
curl localhost:12347/search_v1?query=google&topk=5
```

You will get the following response:

```json
{
  "query": "google",
  "total_results": 5,
  "results": [
    {
      "id": 1,
      "title": "Google",
      "url": "https://www.google.com/",
      "site_name": "www.google.com",
      "date": "",
      "snippet": "Search the world's information, including webpages, images, videos and more. Google has many special features to help you find exactly what you're looking ...",
      "context": "..."
    },
    ...
  ]
}
```

Along with the web search service, we highly recommend organizing your search cache in a local data warehouse. In this project, we store cached web pages in Elasticsearch (ES). As your ES cache grows, you can perform searches not only via web search but also directly over your local cache, which can significantly improve both efficiency and search quality.

### Setting Up Elasticsearch

1. **Download and Start Elasticsearch**

   Download Elasticsearch from the [official website](https://www.elastic.co/downloads/elasticsearch). After extracting, start the ES service:

   ```bash
   ./bin/elasticsearch
   ```

2. **Build and Populate the ES Index**

   Check the script at `tools/es/build_index.py`. On first use, you should run the `create_index()` function to initialize the index. Afterwards, you can index cached web pages into ES as they are collected.

3. **Hybrid Search with Embeddings**

   We use hybrid search in this project, combining traditional keyword search with dense retrieval using embeddings. For embedding, we use the `bge-m3` model. You need to start a vLLM server with this model:

   ```bash
   vllm serve ... --port 25883
   ```

   (Replace `...` with the appropriate model path and options.)

4. **Start the Local ES Search Service**

   Launch the cache-based search service:

   ```bash
   python -m tools.es.cache_search_app
   ```

   This will provide a search API powered by your local Elasticsearch instance, leveraging both keyword and embedding-based retrieval for improved results.

By combining web search with a robust local cache in Elasticsearch and hybrid search techniques, you can achieve faster and more accurate search results for deep search tasks.

A case:

```bash
curl localhost:12348/search -X POST -H "Content-Type: application/json" -d '{"query": "nature", "topk": 5}'
```

You will get the following response:

```json
{
  "results": [
    {
      "content": "Published Time: Wed, 04 Jun 2025 12:01:25 GMT\n\n# WRITING NATURE",
      "score": 1.5914185,
      "snippet": "the poem you wanted, so I hope you will forgive me for sending it to ... ThE PoET's PAEAN To ThE WATER cyclE, TURNEd To. A PhoToGRAPhER's cElEBRATIoN ...",
      "title": "WRITING NATURE",
      "url": "https://aba.org.uk/assets/catalogues/215_DPS.pdf"
    }
  ],
  ...
}
```
In addition to the `/search` route, the `tools/es/cache_search_app.py` service also provides an `/insert` route. This allows you to add new webpages to the Elasticsearch index by sending a POST request with the webpage data. 

Besides, you can also manually call `add_webpages_to_index()` function in `tools/es/build_index.py` to add new webpages to the Elasticsearch index.

> **Tip:** The infrastructure described here is just one possible implementation used in my own project, mainly because deep search tasks require frequent use of these tools and rely heavily on web search. If you have a better solution or a different setup that fits your needs, feel free to replace or modify this part. This infrastructure is entirely optional and not required for all use cases.

## :notebook: Quick Start
To quickly evaluate MetaAgent, please refer to the technical report for detailed instructions. You can run the evaluation with the following command:

```bash
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
```

Make sure you have the following services running:

- Qwen2.5-7B-Instruct: http://localhost:12346/v1/
- QwQ-32B: http://localhost:12345/v1/
- Search Service: http://localhost:12347/search_v1
- Cache Search Service: http://localhost:12348/search
- MySQL: http://localhost:3306
- Elasticsearch: http://localhost:9200

The Qwen models are hosted using VLLM:
```bash
vllm serve qwen2.5-7b-instruct --port 12346
vllm serve QwQ-32B --port 12345
```

You can use other OpenAI-style models by replacing the `--reasoning_model` and `--auxiliary_model` arguments with the model name.
Besides, if advanced reasoning model is specified, MetaAgent will use the advanced reasoning model as the central reasoning model.
> **Tip:** Check all ports in these services to enable communication between the services.

In the data folder, we provide the GAIA subset and WebWalker dataset we use in this project. Another used dataset is BrowseCamp, of which the author require not release decoded version in public place, there you may download and decode by yourself.



## :raised_hands: FAQs
 
 If you have any questions, please feel free to contact me at `tommy[at]chien.io`.

### Citation:
```bibtex
@article{qian2025metaagent,
  title={MetaAgent: Toward Self-Evolving Agent via Tool Meta-Learning},
  author={Hongjin Qian and Zheng Liu},
  year={2025}
}
```

