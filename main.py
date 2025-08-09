# main.py
from fastapi import FastAPI, Header, HTTPException, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, HttpUrl
from typing import List

# Import the core logic from our processor file
from processor import process_document_and_questions

# --- API Setup ---
app = FastAPI(
    title="HackRx Document Processing API",
    description="API to answer questions based on a document URL using Pinecone, Postgres, and Ollama.",
)

# --- Authentication ---
# IMPORTANT: Replace "YOUR_SECRET_API_KEY" with the real key from the platform.
API_KEY = "YOUR_SECRET_API_KEY"
API_KEY_NAME = "Authorization"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)

async def get_api_key(api_key_header: str = Security(api_key_header)):
    """Validates the Bearer token."""
    if api_key_header == f"Bearer {API_KEY}":
        return api_key_header
    else:
        raise HTTPException(status_code=403, detail="Could not validate credentials")

# --- Pydantic Models for Request/Response ---
class RequestPayload(BaseModel):
    documents: HttpUrl
    questions: List[str]

class ResponsePayload(BaseModel):
    answers: List[str]

# --- API Endpoint ---
@app.post("/hackrx/run", response_model=ResponsePayload)
async def run_hackrx(
    payload: RequestPayload,
    api_key: str = Security(get_api_key)
):
    """
    This endpoint receives a document URL and a list of questions,
    and returns a list of answers derived from the document.
    """
    print(f"Received request for document: {payload.documents}")

    # Call the processing function from processor.py
    answers = process_document_and_questions(
        pdf_url=str(payload.documents),
        questions=payload.questions
    )

    print("Request complete. Returning answers.")
    return {"answers": answers}

# --- Root endpoint for health check ---
@app.get("/")
def read_root():
    return {"status": "API is running"}

# To run the server locally: uvicorn main:app --reload