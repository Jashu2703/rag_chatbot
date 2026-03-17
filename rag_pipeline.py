import os
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

# Load environment variables
load_dotenv()
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
os.environ["HF_TOKEN"] = HUGGINGFACE_API_TOKEN

def main():
    # Data Ingestion
    url = "https://www.educosys.com/"
    loader = WebBaseLoader(url)
    documents = loader.load()

    # Text Processing
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    # Vectorization & Storage
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma.from_documents(texts, embeddings)

    # Language Model
    # Note: google/flan-t5-xxl might cause issues. Consider using a different model like 'microsoft/DialoGPT-medium' or 'google/flan-t5-base'
    llm = HuggingFaceEndpoint(
        repo_id="microsoft/DialoGPT-medium",
        huggingfacehub_api_token=HUGGINGFACE_API_TOKEN,
        temperature=0.7,
        max_new_tokens=512
    )

    # Example query
    query = input("Enter your question about Educosys: ")

    # RAG Chain
    retriever = db.as_retriever()
    docs = retriever.invoke(query)
    context = "\n".join([doc.page_content for doc in docs])
    print("Retrieved context:")
    print(context)
    print("\nQuery:", query)
    print("Note: LLM generation is commented out due to API issues. To enable, uncomment the LLM code and ensure a valid Hugging Face token.")

if __name__ == "__main__":
    main()