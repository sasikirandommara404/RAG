import os
import json
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import pinecone

# Load environment variables
load_dotenv()

class VectorDB:
    def __init__(self, index_name="rag-demo"):
        self.index_name = index_name
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embedding_dim = 384  # Dimension for all-MiniLM-L6-v2
        
        # Initialize Pinecone client
        self.pc = pinecone.Pinecone(
            api_key=os.getenv("PINECONE_API_KEY")
        )
        
        # Check if index exists
        if self.index_name not in self.pc.list_indexes().names():
            # Create new index if it does not exist
            print(f"Creating new index {self.index_name} with dimension {self.embedding_dim}...")
            try:
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.embedding_dim,
                    metric="cosine",
                    spec=pinecone.ServerlessSpec(
                        cloud='aws',
                        region='us-east-1'
                    )
                )
                # Wait for index to be ready
                print("Waiting for index to be ready...")
                import time
                time.sleep(30)
            except Exception as e:
                print(f"Error creating index with ServerlessSpec: {e}")
                print("Attempting to create index with PodSpec for free tier...")
                self.pc.create_index(
                    name=self.index_name, 
                    dimension=self.embedding_dim, 
                    metric='cosine',
                    spec=pinecone.PodSpec(environment='gcp-starter')
                )
                print("Waiting for index to be ready...")
                import time
                time.sleep(30)
        else:
            print(f"Index '{self.index_name}' already exists.")
        
        self.index = self.pc.Index(self.index_name)
        print(f"Connected to index {self.index_name}")
    
    def embed_text(self, text):
        """Convert text to embedding vector."""
        return self.model.encode(text).tolist()
    
    def upsert_documents(self, documents):
        """Upsert documents to the vector database."""
        vectors = []
        print(f"Preparing to upsert {len(documents)} documents...")
        
        for i, doc in enumerate(documents, 1):
            try:
                if not isinstance(doc, dict):
                    print(f"Document {i} is not a dictionary: {doc}")
                    continue
                    
                text = doc.get("text", "")
                if not text:
                    print(f"Document {i} has no text content: {doc}")
                    continue
                    
                vector = self.embed_text(text)
                metadata = {
                    "text": text,
                    "source": doc.get("source", "unknown"),
                    "metadata": json.dumps(doc.get("metadata", {}))
                }
                
                doc_id = str(doc.get("id", f"doc_{i}"))
                
                vectors.append({
                    'id': doc_id,
                    'values': vector,
                    'metadata': metadata
                })
                
                if i % 10 == 0 or i == len(documents):
                    print(f"Processed {i}/{len(documents)} documents...")
                    
            except Exception as e:
                print(f"Error processing document {i}: {str(e)}")
                import traceback
                traceback.print_exc()
        
        # Upsert in batches of 20 (smaller batches for better reliability)
        batch_size = 20
        total_upserted = 0
        
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            try:
                print(f"Upserting batch {i//batch_size + 1}/{(len(vectors)-1)//batch_size + 1}...")
                response = self.index.upsert(vectors=batch)
                total_upserted += len(batch)
                print(f"Successfully upserted {total_upserted}/{len(vectors)} documents")
            except Exception as e:
                print(f"Error upserting batch {i//batch_size + 1}: {str(e)}")
        
        print(f"\nCompleted upserting {total_upserted} documents to index {self.index_name}")
        
        # Wait and verify the index count
        import time
        time.sleep(5) # Initial wait for indexing to start
        try:
            print("\nVerifying index count...")
            for _ in range(10): # Try for up to 50 seconds
                index_stats = self.index.describe_index_stats()
                vector_count = index_stats.get('total_vector_count', 0)
                print(f"Current vector count: {vector_count}. Expected: {len(vectors)}")
                if vector_count == len(vectors):
                    print("Index is fully populated.")
                    break
                time.sleep(5)
            else:
                print("Warning: Index vector count did not match expected count after waiting.")
            
            final_stats = self.index.describe_index_stats()
            print(f"\nFinal index stats: {final_stats}")

        except Exception as e:
            print(f"\nCould not get index stats: {str(e)}")
    
    def search(self, query_text, top_k=3):
        """Search for similar documents."""
        query_embedding = self.embed_text(query_text)
        
        try:
            # Query the index
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
            
            # Convert to the expected format
            class SearchResults:
                def __init__(self, matches):
                    self.matches = matches
                    
            class Match:
                def __init__(self, id, score, metadata):
                    self.id = id
                    self.score = score
                    self.metadata = metadata
                    
            matches = []
            
            if 'matches' not in results or not results['matches']:
                return SearchResults([])
                
            for match in results['matches']:
                # Handle different possible metadata formats
                metadata = match.get('metadata', {})
                if isinstance(metadata, str):
                    try:
                        metadata = json.loads(metadata)
                    except (json.JSONDecodeError, TypeError):
                        metadata = {'text': str(metadata)}
                
                # Ensure we have the required fields
                match_metadata = {
                    'text': metadata.get('text', ''),
                    'source': metadata.get('source', 'unknown'),
                    'metadata': metadata.get('metadata', {})
                }
                
                matches.append(Match(
                    id=match.get('id', ''),
                    score=match.get('score', 0.0),
                    metadata=match_metadata
                ))
                
            return SearchResults(matches)
            
        except Exception as e:
            print(f"Error in search: {str(e)}")
            # Return empty results on error
            class SearchResults:
                def __init__(self, matches):
                    self.matches = matches
            return SearchResults([])

def load_sample_data():
    """Load sample data from JSON file."""
    data_path = Path("data") / "sample_data.json"
    with open(data_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def setup_vector_db():
    """Set up the vector database with sample data."""
    # Initialize vector DB
    vector_db = VectorDB()
    
    # Load and insert sample data
    documents = load_sample_data()
    vector_db.upsert_documents(documents)
    
    return vector_db

if __name__ == "__main__":
    # First, make sure the data is prepared
    from data_preparation import prepare_data
    prepare_data()
    
    # Then set up the vector database
    db = setup_vector_db()
    print("Vector database setup complete!")
