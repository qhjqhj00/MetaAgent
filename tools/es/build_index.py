from elasticsearch import Elasticsearch
import requests
import json
import asyncio
import tiktoken

def get_tokenizer(model: str = "gpt-3.5-turbo"):
    return tiktoken.encoding_for_model(model)

def load_jsonl(file_path):
    with open(file_path, "r") as f:
        return [json.loads(line) for line in f]

tokenizer = get_tokenizer()
es = Elasticsearch("http://localhost:9311")

import aiohttp
from openai import AsyncOpenAI

async def get_embedding_from_vllm(text):
    """Get embedding from vllm service running on localhost:8000"""
    client = AsyncOpenAI(
        api_key="not-needed",
        base_url="http://localhost:25883/v1/"
    )
    
    response = await client.embeddings.create(
            input=text,
            model="bge-m3" # The model name will be det ermined by what's running on vLLM
        )
    return response.data[0].embedding

def create_index(index_name="webpage"):
    """Create Elasticsearch index with mappings
    
    Args:
        index_name: Name of the ES index to create
    """
    # Create index with mappings
    mapping = {
        "mappings": {
            "properties": {
                "url": {"type": "keyword"},
                "snippet": {"type": "text"},
                "title": {"type": "text"}, 
                "content": {"type": "text"},
                "content_embedding": {"type": "dense_vector", "dims": 1024}
            }
        }
    }

    if es.indices.exists(index=index_name):
        es.indices.delete(index=index_name)
    es.indices.create(index=index_name, body=mapping)

def truncate_text(text, max_length=1024):
    """Truncate text to max_length"""
    return tokenizer.decode(tokenizer.encode(text)[:max_length])

async def index_documents(data, index_name="webpage"):
    """Index documents into Elasticsearch with embeddings
    
    Args:
        data: List of dicts with keys: url, snippet, title, content
        index_name: Name of the ES index to index into
    """
    # Process in batches of 100 documents
    batch_size = 25
    for i in range(0, len(data), batch_size):
        print(f"Processing batch {i//batch_size + 1} of {len(data)//batch_size}")
        batch = data[i:i+batch_size]
        
        # Get embeddings for current batch
        valid_data = []
        embedding_tasks = []
        for item in batch:
            try:
                embedding_tasks.append(get_embedding_from_vllm(truncate_text(item["content"])))
                valid_data.append(item)
            except Exception as e:
                print(f"Error processing item: {e}")
                continue
                
        embeddings = await asyncio.gather(*embedding_tasks)

        # Build bulk data for current batch
        bulk_data = []
        for doc, embedding in zip(valid_data, embeddings):
            doc["content_embedding"] = embedding
            bulk_data.append({"index": {"_index": index_name}})
            bulk_data.append(doc)
        
        # Bulk index current batch
        es.bulk(body=bulk_data)
        es.indices.refresh(index=index_name)

async def add_webpages_to_index(webpages, index_name="webpage"):
    """Add a list of webpages to the Elasticsearch index
    
    Args:
        webpages: List of dicts with keys: url, title, snippet, content
        index_name: Name of the ES index to add to (defaults to "webpage")
    """
    if not webpages:
        print("No webpages to add")
        return
        
    print(f"Adding {len(webpages)} webpages to index '{index_name}'")
    
    # Check existing URLs in the index
    existing_urls = set()
    try:
        # Get all existing URLs using scroll API
        query = {
            "query": {"match_all": {}},
            "_source": ["url"]
        }
        scroll_response = es.search(
            index=index_name,
            body=query,
            scroll='2m',
            size=1000
        )
        
        while True:
            for hit in scroll_response['hits']['hits']:
                existing_urls.add(hit['_source']['url'])
            
            scroll_id = scroll_response['_scroll_id']
            scroll_response = es.scroll(scroll_id=scroll_id, scroll='2m')
            
            if not scroll_response['hits']['hits']:
                break
                
        print(f"Found {len(existing_urls)} existing URLs in index")
    except Exception as e:
        print(f"Error checking existing URLs (index might not exist): {e}")
        existing_urls = set()
    
    # Filter out webpages with existing URLs
    new_webpages = []
    skipped_count = 0
    for webpage in webpages:
        if webpage.get("url") in existing_urls:
            skipped_count += 1
            continue
        existing_urls.add(webpage.get("url"))
        new_webpages.append(webpage)
    with open("data/existing_urls.txt", "w") as f:
        for url in existing_urls:
            f.write(url + "\n")
    print(f"Skipped {skipped_count} webpages with existing URLs")
    print(f"Processing {len(new_webpages)} new webpages")
    
    if not new_webpages:
        print("No new webpages to add")
        return
    
    # Process in batches
    batch_size = 100
    total_processed = 0
    
    for i in range(0, len(new_webpages), batch_size):
        batch = new_webpages[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1} of {(len(new_webpages)-1)//batch_size + 1}")
        
        # Get embeddings for current batch
        valid_data = []
        embedding_tasks = []
        
        for webpage in batch:
            try:
                # Validate required keys
                if not all(key in webpage for key in ["url", "title", "snippet", "content"]):
                    print(f"Skipping webpage missing required keys: {webpage.get('url', 'unknown')}")
                    continue
                    
                embedding_tasks.append(get_embedding_from_vllm(truncate_text(webpage["content"])))
                valid_data.append(webpage)
            except Exception as e:
                print(f"Error processing webpage {webpage.get('url', 'unknown')}: {e}")
                continue
        
        if not valid_data:
            continue
            
        # Get embeddings
        try:
            embeddings = await asyncio.gather(*embedding_tasks)
        except Exception as e:
            print(f"Error getting embeddings for batch: {e}")
            continue

        # Build bulk data for current batch
        bulk_data = []
        for webpage, embedding in zip(valid_data, embeddings):
            webpage_doc = webpage.copy()
            webpage_doc["content_embedding"] = embedding
            bulk_data.append({"index": {"_index": index_name}})
            bulk_data.append(webpage_doc)
            # print(webpage_doc)
        
        # Bulk index current batch
        try:
            response = es.bulk(body=bulk_data)
            print(response)
            if response.get("errors"):
                print(f"Some documents failed to index in batch {i//batch_size + 1}")
            total_processed += len(valid_data)
            print(f"Successfully indexed {len(valid_data)} webpages (total: {total_processed})")
        except Exception as e:
            print(f"Error bulk indexing batch: {e}")
            continue
    
    # Refresh index
    try:
        es.indices.refresh(index=index_name)
        print(f"Index refresh completed. Total new webpages added: {total_processed}")
    except Exception as e:
        print(f"Error refreshing index: {e}")



if __name__ == "__main__":
    # create index for the first time
     # create_index()

    # add cached data to index
    data = load_jsonl('data/cached.jsonl')
   
    asyncio.run(add_webpages_to_index(data))
