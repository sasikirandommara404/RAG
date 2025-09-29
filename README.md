# RAG Pipeline with Pinecone and Gemini

This project implements a Retrieval-Augmented Generation (RAG) pipeline using Pinecone as the vector database and Google's Gemini for text generation.

## Prerequisites

1. Python 3.8+
2. Pinecone account (get API key and environment from [Pinecone Console](https://app.pinecone.io/))
3. Google AI API key (get from [Google AI Studio](https://makersuite.google.com/))

## Setup

1. Clone the repository and navigate to the project directory.

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root and add your API keys:
   ```
   # Pinecone API Configuration
   PINECONE_API_KEY=your_pinecone_api_key
   PINECONE_ENVIRONMENT=your_pinecone_environment
   
   # Google Gemini API Configuration
   GEMINI_API_KEY=your_gemini_api_key
   ```

## Usage

1. First, prepare the sample data:
   ```bash
   python data_preparation.py
   ```

2. Set up the vector database with sample data:
   ```bash
   python vector_db.py
   ```

3. Test the RAG pipeline:
   ```bash
   python rag_pipeline.py
   ```

## Project Structure

- `data_preparation.py`: Script to generate and save sample data
- `vector_db.py`: Handles vector database operations using Pinecone
- `rag_pipeline.py`: Implements the RAG pipeline using Gemini
- `requirements.txt`: Lists all required Python packages
- `.env`: Stores API keys and configuration (not version controlled)

## Customization

- To use your own data, modify the `sample_data` list in `data_preparation.py`
- Adjust the embedding model in `vector_db.py` if needed
- Modify the prompt template in `rag_pipeline.py` to change how the context is used


