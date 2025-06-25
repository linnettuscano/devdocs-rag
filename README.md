# devdocs-rag

A local Retrieval-Augmented Generation (RAG) system combining a local language model with Qdrant vector search for document retrieval, wrapped in an easy-to-use Streamlit interface.

## Overview

This project enables semantic search and question answering over your own document collection using:

- **Qdrant**: A vector search engine for efficient similarity search
- **HuggingFace Transformers**: Local language model (e.g., Flan-T5) for generating answers
- **Streamlit**: Web UI to interact with the system easily
- **Sentence Transformers**: For embedding documents and queries into vectors

## Features

- Search documents using vector similarity
- Answer queries with a local transformer model
- Interactive UI accessible at `http://localhost:8501`
- Supports CPU and Apple MPS device for acceleration

## Getting Started

1. Clone the repository:

   ```bash
   git clone https://github.com/linnettuscano/devdocs-rag.git
   cd devdocs-rag
2. Setup Python environment and install dependencies:
   ```bash
   conda create -n devdocs-rag python=3.10 -y
   conda activate devdocs-rag
   pip install -r requirements.txt
3. Run Qdrant locally (using Docker recommended):
   ```bash
   docker run -p 6333:6333 qdrant/qdrant
4. Start the Streamlit app:
   ```bash
   streamlit run chat_ui.py
5. Open your browser at http://localhost:8501 and start asking questions!
   
## Notes


Make sure your Qdrant collection is populated with your document embeddings before querying.

The project uses HuggingFace’s google/flan-t5-small model locally.

Adjust device settings to use CPU or Apple MPS as per your machine.

## License
This project is licensed under the MIT License.
