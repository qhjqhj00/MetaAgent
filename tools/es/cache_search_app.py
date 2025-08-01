from flask import Flask, request, jsonify
from elasticsearch import Elasticsearch
import asyncio
from openai import AsyncOpenAI
import os
from my_own_tools import *

app = Flask(__name__)

# Initialize Elasticsearch client
es = Elasticsearch("http://localhost:9311")

existing_urls = set()
with open('data/existing_urls.txt', 'r') as f:
    for line in f:
        existing_urls.add(line.strip())

# Initialize tokenizer
tokenizer = get_tokenizer()

async def get_embedding_from_vllm(text):
    """Get embedding from vllm service running on localhost:25883"""
    client = AsyncOpenAI(
        api_key="not-needed",
        base_url="http://localhost:25883/v1"
    )
    
    response = await client.embeddings.create(
        input=[text],
        model="bge-m3"
    )
    
    return response.data[0].embedding

def truncate_text(text, max_length=1024):
    """Truncate text to max_length"""
    return tokenizer.decode(tokenizer.encode(text)[:max_length])

@app.route('/search', methods=['POST'])
def search():
    """Search service that retrieves top 10 results from ES webpage index"""
    try:
        data = request.get_json()
        query = data.get('query', '')
        topk = data.get('topk', 10)
        
        if not query:
            return jsonify({'error': 'Query is required'}), 400
        
        # Get embedding for the query
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        query_embedding = loop.run_until_complete(get_embedding_from_vllm(truncate_text(query)))
        loop.close()
        
        # Search in Elasticsearch using vector similarity
        search_body = {
            "query": {
                "script_score": {
                    "query": {
                        "bool": {
                            "should": [
                                {
                                    "multi_match": {
                                        "query": query,
                                        "fields": ["title^2", "content"],
                                        "type": "best_fields"
                                    }
                                }
                            ]
                        }
                    },
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'content_embedding') + 1.0",
                        "params": {"query_vector": query_embedding}
                    }
                }
            },
            "size": topk,
            "_source": {
                "excludes": ["content_embedding"]
            } 
        }
        
        response = es.search(index="webpage", body=search_body)
        
        results = []
        for hit in response['hits']['hits']:
            source = hit['_source']
            result = {
                'title': source.get('title', ''),
                'url': source.get('url', ''),
                'snippet': source.get('snippet', ''),
                'content': source.get('content', ''),
                'score': hit['_score']
            }
            results.append(result)
        
        return jsonify({'results': results})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/insert', methods=['POST'])
def insert():
    """Insert service that adds new webpages to ES if URL doesn't exist"""
    try:
        data = request.get_json()
        webpages = data.get('webpages', [])
        
        if not webpages:
            return jsonify({'error': 'Webpages list is required'}), 400
        
        # Filter new webpages
        new_webpages = []
        skipped_count = 0
        
        for webpage in webpages:
            # Validate required fields
            if not all(key in webpage for key in ['title', 'url', 'snippet', 'content']):
                continue
                
            if webpage['url'] in existing_urls:
                skipped_count += 1
                continue
                
            new_webpages.append(webpage)
            existing_urls.add(webpage['url'])
        
        if not new_webpages:
            return jsonify({
                'message': 'No new webpages to insert',
                'inserted_count': 0,
                'skipped_count': skipped_count
            })
        
        # Get embeddings and insert into ES
        bulk_data = []
        successful_urls = []
        
        # Create event loop for async operations
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            for webpage in new_webpages:
                try:
                    # Get embedding for content
                    embedding = loop.run_until_complete(get_embedding_from_vllm(truncate_text(webpage['content'])))

                    # Prepare document for ES
                    doc = {
                        'title': webpage['title'],
                        'url': webpage['url'],
                        'snippet': webpage['snippet'],
                        'content': webpage['content'],
                        'content_embedding': embedding
                    }
                    
                    bulk_data.append({"index": {"_index": "webpage", "_id": webpage['url']}})
                    bulk_data.append(doc)
                    successful_urls.append(webpage['url'])
                    
                except Exception as e:
                    print(f"Error processing webpage {webpage['url']}: {e}")
                    continue
            
            # Bulk insert into ES
            inserted_count = 0
            if bulk_data:
                response = es.bulk(body=bulk_data)
                if not response.get("errors"):
                    inserted_count = len(successful_urls)
                    # Refresh index
                    es.indices.refresh(index="webpage")
                    
                else:
                    return jsonify({'error': 'Failed to insert some documents'}), 500
            
            return jsonify({
                'message': f'Successfully inserted {inserted_count} webpages',
                'inserted_count': inserted_count,
                'skipped_count': skipped_count
            })
            
        finally:
            loop.close()
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=39118)
