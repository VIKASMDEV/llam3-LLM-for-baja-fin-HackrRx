# processor.py
import os
import requests
import fitz  # PyMuPDF
from dotenv import load_dotenv

# LangChain Imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate

# Database Imports
import psycopg2
from psycopg2 import sql

# Load environment variables from .env file
load_dotenv()

# --- Component & Client Initialization ---
# Pinecone
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "hackrx-index"  # The name of your index in Pinecone

# PostgreSQL
DB_CONNECTION_STRING = "postgresql://user:password@localhost:5432/vector_db"

# LangChain
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = Ollama(model="llama3.1:8b-instruct")

ANSWER_PROMPT = PromptTemplate.from_template(
    """You are an expert at finding answers in a document.
    Answer the following question based ONLY on the provided context.
    If the answer is not in the context, state that the answer could not be found.
    Be concise and extract the answer directly from the text.

    Context: {context}
    Question: {question}
    Answer: """
)


# --- Database Helper Functions ---

def setup_database():
    """Ensures the document tracking table exists."""
    conn = psycopg2.connect(DB_CONNECTION_STRING)
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS processed_documents (
                id SERIAL PRIMARY KEY,
                document_url TEXT UNIQUE NOT NULL,
                indexed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );
        """)
    conn.commit()
    conn.close()


def is_document_processed(document_url: str) -> bool:
    """Checks if a document URL has already been processed."""
    conn = psycopg2.connect(DB_CONNECTION_STRING)
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM processed_documents WHERE document_url = %s;", (document_url,))
        exists = cur.fetchone() is not None
    conn.close()
    return exists


def mark_document_as_processed(document_url: str):
    """Adds a document URL to the tracking table."""
    conn = psycopg2.connect(DB_CONNECTION_STRING)
    with conn.cursor() as cur:
        cur.execute("INSERT INTO processed_documents (document_url) VALUES (%s) ON CONFLICT (document_url) DO NOTHING;",
                    (document_url,))
    conn.commit()
    conn.close()


# --- Main Processing Function ---

def process_document_and_questions(pdf_url: str, questions: list[str]) -> list[str]:
    """
    Orchestrates the entire process: check cache, ingest to Pinecone if new, and answer questions.
    """
    try:
        # Check if document needs processing
        if not is_document_processed(pdf_url):
            print(f"New document detected. Processing and indexing: {pdf_url}")

            # 1. Download and Load PDF
            response = requests.get(pdf_url)
            response.raise_for_status()
            pdf_bytes = response.content
            doc_text = ""
            with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
                for page in doc:
                    doc_text += page.get_text()

            # 2. Chunk text
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = text_splitter.split_text(doc_text)

            # 3. Index in Pinecone using a namespace
            PineconeVectorStore.from_texts(
                texts=chunks,
                embedding=embeddings,
                index_name=PINECONE_INDEX_NAME,
                namespace=pdf_url  # Use the URL as a unique namespace
            )

            # 4. Mark as processed in PostgreSQL
            mark_document_as_processed(pdf_url)
            print("Indexing complete.")

        else:
            print(f"Document already processed. Retrieving from cache: {pdf_url}")

        # 5. Initialize vector store for querying
        vector_store = PineconeVectorStore.from_existing_index(
            index_name=PINECONE_INDEX_NAME,
            embedding=embeddings,
            namespace=pdf_url
        )
        retriever = vector_store.as_retriever(search_kwargs={'k': 3})
        answer_chain = ANSWER_PROMPT | llm

        answers = []
        for i, question in enumerate(questions):
            print(f"Answering question {i + 1}/{len(questions)}: {question}")
            retrieved_docs = retriever.invoke(question)
            context = "\n---\n".join([doc.page_content for doc in retrieved_docs])
            answer = answer_chain.invoke({"context": context, "question": question})
            answers.append(answer.strip())

        return answers

    except Exception as e:
        print(f"An error occurred in processor: {e}")
        return [f"An unexpected error occurred: {e}"] * len(questions)


# Call setup_database once on startup
setup_database()