from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Qdrant
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# Load embeddings
embeddings = OpenAIEmbeddings()

# Connect to Qdrant
vectordb = Qdrant(
    url="http://vectordb:6333",
    collection_name="devdocs",
    embeddings=embeddings,
)

# Set up retrieval + generation chain
retriever = vectordb.as_retriever(search_kwargs={"k": 4})
llm = ChatOpenAI(temperature=0.2)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

def get_answer(question: str) -> str:
    result = qa_chain.run(question)
    return result
