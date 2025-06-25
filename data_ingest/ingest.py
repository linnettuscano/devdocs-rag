import os
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from dotenv import load_dotenv

load_dotenv()  # Load OPENAI_API_KEY from .env

# Load all .md or .txt docs from ./docs
loader = DirectoryLoader("./docs", glob="**/*.md", show_progress=True)
docs = loader.load()

# Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# Create embeddings
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Initialize Qdrant connection
qdrant = Qdrant.from_documents(
    documents=chunks,
    embedding=embedding,
    url="http://localhost:6333",
    collection_name="devdocs"
)

print(f"Ingested {len(chunks)} chunks into Qdrant.")
