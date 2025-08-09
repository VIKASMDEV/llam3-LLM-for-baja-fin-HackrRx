# LLM Document Processing System

This repository contains the source code for a Retrieval-Augmented Generation (RAG) API built for the HackRx competition. The application processes PDF documents from a URL, indexes their content, and answers user questions based on the information contained within the document.

## Features

-   **Dynamic Document Processing**: Ingests and processes PDFs directly from a URL.
-   **RAG Pipeline**: Uses a modern RAG architecture to provide accurate, context-aware answers.
-   **Multi-Layered Output**: Provides both a raw JSON output for system use and a friendly, conversational response for end-users.
-   **Scalable Stack**: Built with a professional tech stack including FastAPI, Pinecone, and PostgreSQL.

## Tech Stack

-   **Backend**: FastAPI
-   **LLM**: Ollama (`Llama 3.1 8B-Instruct`)
-   **Vector Database**: Pinecone
-   **Metadata & Caching**: PostgreSQL
-   **Containerization**: Docker
-   **Core AI Framework**: LangChain

## Getting Started

Follow these instructions to set up and run the project locally.
go throught requirements.txt install the prerequisites

### Prerequisites

-   [Git](https://git-scm.com/)
-   [Python 3.11+](https://www.python.org/)
-   [Docker](https://www.docker.com/products/docker-desktop/) and Docker Compose
-   [Ollama](https://ollama.com/) installed with a model pulled (e.g., `ollama pull llama3.1:8b-instruct`)
-   A [Pinecone](https://www.pinecone.io/) account and API key.

### 1. Clone the Repository

```bash
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name
