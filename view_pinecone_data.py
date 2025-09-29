import os
from dotenv import load_dotenv
from pinecone import Pinecone
import json

def view_data_in_pinecone(index_name="rag-demo"):
    """Fetches and displays all documents from a Pinecone index."""
    load_dotenv()

    # 1. Initialize Pinecone
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        print("PINECONE_API_KEY not found in .env file.")
        return

    pc = Pinecone(api_key=api_key)

    # 2. Connect to the index
    if index_name not in pc.list_indexes().names():
        print(f"Index '{index_name}' does not exist.")
        return
    
    index = pc.Index(index_name)
    print(f"Successfully connected to index '{index_name}'.")

    # 3. Get index statistics
    stats = index.describe_index_stats()
    print(f"\nIndex Statistics:\n{stats}")

    total_vectors = stats.get('total_vector_count', 0)
    if total_vectors == 0:
        print("The index is empty.")
        return

    # 4. Fetch all vectors
    # Pinecone's query-all approach: query with a zero vector to get all items
    print("\nFetching all vectors from the index...")
    try:
        # Note: A more robust way for large indexes is pagination
        # For this small example, querying for all is fine.
        all_vectors = index.query(
            vector=[0]*384, # Query with a dummy zero vector
            top_k=total_vectors, # Ask for all vectors
            include_metadata=True
        )

        print(f"\n--- Stored Documents (Total: {len(all_vectors['matches'])}) ---")
        for i, match in enumerate(all_vectors['matches']):
            metadata = match.get('metadata', {})
            print(f"\n--- Document {i+1} ---")
            print(f"  ID: {match.get('id')}")
            print(f"  Score (relative to zero vector): {match.get('score'):.4f}")
            print(f"  Source: {metadata.get('source')}")
            print(f"  Text: {metadata.get('text')}")
        print("\n-----------------------------------------")

    except Exception as e:
        print(f"An error occurred while fetching vectors: {e}")

if __name__ == "__main__":
    view_data_in_pinecone()
