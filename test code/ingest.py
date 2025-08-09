# ingest_postgres.py
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres.vectorstores import PGVector

# --- Configuration ---
SOURCE_DOCS_PATH = "../source_documents/"
COLLECTION_NAME = "insurance_policies" # This will be the table name in PostgreSQL

# --- PostgreSQL Connection String ---
# Format: "postgresql+psycopg2://USER:PASSWORD@HOST:PORT/DB_NAME"
CONNECTION_STRING = "postgresql+psycopg2://user:password@localhost:5432/vector_db"

def ingest_documents():
    """
    Loads documents, splits them, and stores them in PostgreSQL with pgvector.
    """
    print("Loading documents...")
    # This loader specifically looks for PDF files in the source folder
    loader = DirectoryLoader(
        SOURCE_DOCS_PATH,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True,
        use_multithreading=True
    )
    documents = loader.load()

    if not documents:
        print(f"No PDF documents found in '{SOURCE_DOCS_PATH}'. Please add your files.")
        return

    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} text chunks.")

    print("Initializing embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print(f"Connecting to PostgreSQL and ingesting documents into table '{COLLECTION_NAME}'...")
    # This will create the table (if it doesn't exist) and store the embeddings
    PGVector.from_documents(
        embedding=embeddings,
        documents=chunks,
        collection_name=COLLECTION_NAME,
        connection_string=CONNECTION_STRING,
    )
    print("Ingestion complete!")

if __name__ == '__main__':
    ingest_documents()