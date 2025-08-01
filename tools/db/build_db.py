import mysql.connector
import json
from tqdm import tqdm

api_dict = json.load(open("data/api_dict.json"))


def build_db(db_name):
    # Get database connection info from api_dict
    db_config = api_dict["db"][db_name]
    
    # Connect to MySQL server
    conn = mysql.connector.connect(
        host=db_config["host"],
        port=db_config["port"],
        user=db_config["user"],
        password=db_config["password"]
    )
    cursor = conn.cursor()
    cursor.execute("CREATE DATABASE IF NOT EXISTS search;")

    # Create search_results table
    cursor.execute("USE search;")
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS page_results (
            url VARCHAR(512) PRIMARY KEY,
            title TEXT,
            snippet TEXT,
            content LONGTEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            metainfo JSON
        )
    """)
    # Print page_results table length
    cursor.execute("SELECT COUNT(*) FROM page_results;")
    count = cursor.fetchone()[0]
    print(f"page_results table has {count} records")

    # Commit changes and close connection
    conn.commit()
    cursor.close()
    conn.close()

def drop_page_results_table(db_name):
    # Get database connection info from api_dict
    db_config = api_dict["db"][db_name]
    
    # Connect to MySQL server
    conn = mysql.connector.connect(
        host=db_config["host"],
        port=db_config["port"],
        user=db_config["user"],
        password=db_config["password"]
    )
    cursor = conn.cursor()
    
    # Use the search database
    cursor.execute("USE search;")
    cursor.execute("SELECT COUNT(*) FROM page_results;")
    count = cursor.fetchone()[0]
    print(f"page_results table has {count} records")
    # Drop the page_results table
    cursor.execute("DROP TABLE IF EXISTS page_results;")
    
    print("page_results table has been dropped successfully.")
    
    # Commit changes and close connection
    conn.commit()
    cursor.close()
    conn.close()


if __name__ == "__main__":

    # build db for the first time
    # build_db("root")


    # Connect to MySQL database
    db_config = api_dict["db"]["root"]
    conn = mysql.connector.connect(
        host=db_config["host"],
        port=db_config["port"],
        user=db_config["user"],
        password=db_config["password"]
    )
    cursor = conn.cursor()
    cursor.execute("USE search;")
    cursor.execute("SELECT COUNT(*) FROM page_results;")
    count = cursor.fetchone()[0]
    print(f"page_results table has {count} records")
    # print(f"page_results table has {count} records")

    ## store existing data in page_results table
    for line in tqdm(open("all_data.jsonl")):
        data = json.loads(line)
        url = data["url"]
        title = data["title"]
        snippet = data["snippet"]
        content = data["content"]
        
        # Insert data into page_results table with empty metainfo
        try:
            cursor.execute(
                "INSERT INTO page_results (url, title, snippet, content, metainfo) VALUES (%s, %s, %s, %s, %s)",
                (url, title, snippet, content, None)
            )
        except Exception as e:
            print(f"Error inserting data for URL {url}: {e}")
            continue
    cursor.execute("SELECT COUNT(*) FROM page_results;")
    count = cursor.fetchone()[0]
    print(f"page_results table has {count} records")
    # Commit changes and close connection
    conn.commit()
    cursor.close()
    conn.close()
        

