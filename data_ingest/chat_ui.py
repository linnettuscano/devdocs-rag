import streamlit as st
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain.chains import RetrievalQA
from qdrant_client import QdrantClient
from transformers import pipeline

def main():
    st.title("Local LLM + Qdrant Vector Search Demo")

    # Load embeddings model
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Qdrant client (ensure Qdrant is running locally)
    client = QdrantClient(url="http://localhost:6333")

    # Connect to Qdrant collection
    qdrant = QdrantVectorStore(
        client=client,
        collection_name="rag_data",  # This collection must exist
        embedding=embedding,
    )

    # Define the local HuggingFace generation pipeline using flan-t5-base
    hf_pipeline = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        max_length=512,
        do_sample=False,
        device=-1, 
    )

    # Wrap the HF pipeline into LangChain
    llm = HuggingFacePipeline(pipeline=hf_pipeline)

    # Retrieval-based QA
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=qdrant.as_retriever()
    )

    # Input from user
    query = st.text_input("Enter your question:")

    if query:
        response = qa.invoke(query)
        answer = response['result'] if isinstance(response, dict) else response
        st.markdown(f"**Answer:** {answer}")


if __name__ == "__main__":
    main()
