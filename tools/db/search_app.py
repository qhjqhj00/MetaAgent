from tools.search import (
    extract_relevant_info, 
    fetch_page_content,
    extract_snippet_with_context,
    google_search_async 
)   
import asyncio
from typing import List
import json
import mysql.connector
from fastapi import FastAPI, Query
from typing import Optional

api_dict = json.load(open("data/api_dict.json"))

def connect_to_db():
    conn = mysql.connector.connect(
        host=api_dict["db"]["root"]["host"],
        port=api_dict["db"]["root"]["port"],
        user=api_dict["db"]["root"]["user"],
        password=api_dict["db"]["root"]["password"],
        database="search"
    )
    cursor = conn.cursor()
    return conn, cursor

async def web_search(query: str) -> List[str]:
    # Connect to MySQL database
    conn, cursor = connect_to_db()

    # Check if results exist in database
    cursor.execute("SELECT results FROM search_results WHERE query = %s", (query,))
    result = cursor.fetchone()

    if result:
        return extract_relevant_info(json.loads(result[0]))
    
    # If not found, perform Google search
    results = await google_search_async(
        query, 
        api_dict["search_engine"]["google_news"]["api_key"], 
        api_dict["search_engine"]["google_news"]["cse_id"]
    )
    

    # Store results in database
    cursor.execute(
        "INSERT INTO search_results (query, results) VALUES (%s, %s)",
        (query, json.dumps(results))
    )
    conn.commit()
    cursor.close()
    conn.close()
    return extract_relevant_info(results)

async def fetch_and_extract_context(results: dict) -> List[dict]:
    """
    Fetch and extract context for each search result.
    First checks database for existing results, then fetches missing ones.
    
    Args:
        results (dict): Search results from Google/Bing search
        
    Returns:
        List[dict]: List of search results with added context
    """
    # Connect to database
    conn, cursor = connect_to_db()
    
    # Extract URLs and snippets from results
    urls_to_fetch = []
    
    for item in results:
        url = item.get('url')
        if url:
            # Check if URL exists in database
            cursor.execute("SELECT content FROM page_results WHERE url = %s", (url,))
            db_result = cursor.fetchone()
            
            if db_result:
                # Use existing result from database
                item['context'] = db_result[0]
            else:
                # Add to fetch list
                urls_to_fetch.append(url)
    
    if urls_to_fetch:
        # Fetch content for missing URLs
        contents = fetch_page_content(
            urls=urls_to_fetch,
            use_jina=True,
            jina_api_key=api_dict["jina"]["api_key"],
            show_progress=True
        )
        
        # Store new results in database and add context to results
        for item in results:
            url = item.get('url')
            if url in contents:
                item['context'] = contents[url]
                # Store complete item info in database
                try:
                    cursor.execute(
                    "INSERT INTO page_results (url, title, snippet, content, metainfo) VALUES (%s, %s, %s, %s, %s)",
                        (url, item.get('title', ''), item.get('snippet', ''), contents[url], json.dumps(item))
                    )
                    with open("data/incremental_search.jsonl", "a") as f:
                        f.write(json.dumps({
                            "url": url,
                            "title": item.get('title', ''),
                            "snippet": item.get('snippet', ''),
                            "content": contents[url],
                        }) + "\n")
                except Exception as e:
                    print(f"Error inserting page results: {e}")
                    continue
        conn.commit()
    
    cursor.close()
    conn.close()
    return results


app = FastAPI(title="Search API", description="API for web search and content extraction")

@app.get("/search_v1")
async def search(
    query: str = Query(..., description="Search query string"),
    topk: Optional[int] = Query(5, description="Number of results to return", ge=1, le=20)
):
    """
    Perform web search and return results with context
    
    Args:
        query: Search query string
        topk: Number of results to return (default: 5, max: 20)
        
    Returns:
        List of search results with context
    """
    # Perform web search
    search_results = await web_search(query)
    
    # Get top k results
    top_results = search_results[:topk]
    
    # Fetch and extract context for results
    results_with_context = await fetch_and_extract_context(top_results)
    
    return {
        "query": query,
        "total_results": len(results_with_context),
        "results": results_with_context
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=19288)
