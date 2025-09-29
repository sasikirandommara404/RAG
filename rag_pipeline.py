import os
import json
from dotenv import load_dotenv
import google.generativeai as genai
from vector_db import VectorDB

# Load environment variables
load_dotenv()

class RAGPipeline:
    def __init__(self, vector_db):
        """Initialize the RAG pipeline with a vector database instance."""
        self.vector_db = vector_db
        
        # Initialize Gemini with model selection and fallback
        try:
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            
            preferred_model_name = 'gemini-1.5-flash' # Set modern preference
            
            print("\nChecking for available Gemini models...")
            available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
            print(f"Found models: {available_models}")

            if f'models/{preferred_model_name}' in available_models:
                print(f"Preferred model '{preferred_model_name}' is available. Initializing...")
                self.model = genai.GenerativeModel(preferred_model_name)
            else:
                print(f"Warning: Preferred model '{preferred_model_name}' not found.")
                # Find the first suitable fallback model
                if available_models:
                    fallback_model_name = available_models[0].replace('models/', '')
                    print(f"Using first available model as fallback: '{fallback_model_name}'")
                    self.model = genai.GenerativeModel(fallback_model_name)
                else:
                    raise ValueError("No suitable Gemini models found for content generation.")

        except Exception as e:
            print(f"An error occurred during Gemini API initialization: {e}")
            print("Please ensure your GEMINI_API_KEY is correct and you have access to the models.")
            raise
    
    def call_gemini_generate(self, prompt: str) -> str:
        """Call Gemini generate with graceful fallback if the active model is unavailable.
        - If a 404/not found occurs, list models and switch to a valid one, then retry once.
        - Returns the response text or a friendly error message.
        """
        try:
            resp = self.model.generate_content(prompt)
            return getattr(resp, 'text', '').strip()
        except Exception as e:
            msg = str(e).lower()
            if '404' in msg or 'not found' in msg or 'unsupported' in msg:
                # Re-list models and pick a fallback
                try:
                    models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
                    if not models:
                        print("No Gemini models available for generateContent.")
                        return "I'm sorry, no compatible Gemini models are available right now."
                    fallback_model_name = models[0].replace('models/', '')
                    print(f"Active model unavailable. Switching to fallback model: {fallback_model_name}")
                    self.model = genai.GenerativeModel(fallback_model_name)
                    # Retry once
                    resp = self.model.generate_content(prompt)
                    return getattr(resp, 'text', '').strip()
                except Exception as inner_e:
                    print(f"Failed to switch to a fallback model: {inner_e}")
                    return "I'm sorry, the language model is currently unavailable. Please try again later."
            else:
                print(f"Error generating response: {e}")
                return "I'm sorry, I encountered an error while generating a response."
    
    def _format_context(self, search_results):
        """Format search results into a context string."""
        context = ""
        for i, match in enumerate(search_results.matches):
            # Correctly access the metadata attribute of the Match object
            metadata = match.metadata
            context += f"Context {i+1} (Source: {metadata.get('source', 'Unknown')}):\n"
            context += f"{metadata.get('text', '')}\n\n"
        return context.strip()
    
    def generate_response(self, query, top_k=3):
        """Generate a response using RAG."""
        try:
            # Retrieve relevant context
            search_results = self.vector_db.search(query, top_k=top_k)
            context = self._format_context(search_results)
            
            if not context:
                return {
                    "answer": "I couldn't find any relevant information to answer your question.",
                    "sources": []
                }
            
            # Create the prompt with context
            prompt = f"""You are a helpful AI assistant. Use the following context to answer the question at the end. 
            If the context doesn't contain the answer, just say that you don't know, don't try to make up an answer.
            
            Context:
            {context}
            
            Question: {query}
            
            Answer the question based on the context above. If the answer isn't in the context, say "I don't have enough information to answer that question."
            Answer:"""
            
            # Generate response using Gemini with graceful fallback
            answer = self.call_gemini_generate(prompt)
            
            return {
                "answer": answer,
                "sources": [
                    {
                        "text": match.metadata.get("text", ""),
                        "source": match.metadata.get("source", "Unknown"),
                        "score": getattr(match, 'score', 0.0)
                    }
                    for match in getattr(search_results, 'matches', [])
                    if hasattr(match, 'metadata')
                ]
            }
            
        except Exception as e:
            print(f"Error in generate_response: {e}")
            return {
                "answer": "I encountered an error while processing your request. Please try again.",
                "sources": []
            }

def test_rag_pipeline():
    """Test the RAG pipeline with a sample query."""
    try:
        print("Step 1: Preparing and loading sample data...")
        from data_preparation import prepare_data
        from vector_db import load_sample_data
        prepare_data()
        documents = load_sample_data()
        
        print("\nStep 2: Initializing and populating vector database...")
        vector_db = VectorDB()
        
        # Populate the database with sample data
        print("Populating the vector database with sample data...")
        vector_db.upsert_documents(documents)

        # Initialize the RAG pipeline
        print("\nStep 3: Initializing RAG pipeline...")
        rag = RAGPipeline(vector_db)
        
        # Test query
        queries = [
            "What is the theory of relativity?",
            "Who developed quantum mechanics?",
            "What was the Renaissance?"
        ]
        
        for query in queries:
            print(f"\n{'='*80}")
            print(f"Query: {query}")
            
            # Get and display response
            response = rag.generate_response(query)
            print("\nAnswer:", response["answer"])
            
            if response["sources"]:
                print("\nSources:")
                for i, source in enumerate(response["sources"], 1):
                    print(f"{i}. Source: {source.get('source', 'Unknown')}, Score: {source.get('score', 0):.2f}")
                    print(f"   Text: {source.get('text', '')[:200]}..." if len(source.get('text', '')) > 200 else f"   Text: {source.get('text', '')}")
            else:
                print("\nNo sources found for this query.")
                
    except Exception as e:
        print(f"\nError in test_rag_pipeline: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_rag_pipeline()
