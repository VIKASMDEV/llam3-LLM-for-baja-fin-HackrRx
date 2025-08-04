# ingest.py
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Milvus

# --- Configuration ---
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
COLLECTION_NAME = "insurance_policies_db"
SOURCE_DOCS_PATH = "source_documents/"

def ingest_documents():
    """
    Loads documents from a folder, splits them into chunks,
    creates embeddings, and stores them in Milvus.
    """
    print("Loading documents...")
    loader = DirectoryLoader(
        SOURCE_DOCS_PATH,
        glob="**/*",
        loader_cls=PyPDFLoader, # Assuming PDFs for simplicity, can be expanded
        show_progress=True,
        use_multithreading=True
    )
    documents = loader.load()

    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_documents(documents)

    if not chunks:
        print("No documents found to process. Exiting.")
        return

    print(f"Created {len(chunks)} text chunks.")

    print("Initializing embedding model...")
    # Using a free, high-quality model that runs locally
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print("Connecting to Milvus and ingesting documents...")
    Milvus.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        connection_args={"host": MILVUS_HOST, "port": MILVUS_PORT},
    )
    print(f"Successfully ingested documents into Milvus collection: '{COLLECTION_NAME}'")

if __name__ == '__main__':
    ingest_documents()