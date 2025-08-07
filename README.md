Of course, here is a README file for your GitHub repository.

-----

# AI-Powered Insurance Claim Evaluator

This project demonstrates a sophisticated, multi-stage AI system for evaluating insurance claims. It uses a Retrieval-Augmented Generation (RAG) architecture to provide decisions based on specific policy documents. The system first ingests and indexes insurance policy documents into a vector database. Then, it processes a user's claim, retrieves relevant policy clauses, and makes a structured, auditable decision, which is finally translated into a formal response for the policyholder.

## 🚀 Features

  * **Document Ingestion**: Loads and processes PDF or DOCX documents from a source folder.
  * **Vectorization**: Splits documents into chunks and creates embeddings using `HuggingFace` models.
  * **Vector Storage**: Stores and indexes document vectors using **Milvus** for efficient similarity search.
  * **Multi-Stage Processing**:
    1.  **Query Parsing**: Understands the user's plain-language query and extracts key information into a structured JSON format.
    2.  **Contextual Retrieval**: Fetches the most relevant policy clauses from Milvus based on the parsed query.
    3.  **AI-Powered Decision Making**: Uses a Large Language Model (LLM) via **Ollama** to evaluate the claim against the retrieved clauses and generate a structured JSON decision.
    4.  **Formal Response Generation**: Translates the internal JSON decision into a formal, professional response for the end-user.
  * **Local & Open-Source**: Runs entirely on your local machine using powerful open-source tools like Ollama, Milvus, and HuggingFace.

## 🛠️ System Architecture

The project consists of two main Python scripts:

1.  **`ingest.py`**: This script is responsible for the one-time setup of the knowledge base. It scans a designated folder (`source_documents/`), loads all the policy files, splits them into manageable chunks, generates vector embeddings for each chunk, and stores them in a Milvus vector database collection. This process only needs to be run once, or whenever the source policy documents are updated.

2.  **`query_systemCV.py`**: This is the main application logic. It takes a user's query (e.g., "Is my son's dental surgery covered?") and orchestrates the RAG pipeline. It initializes the connection to Milvus and the Ollama LLM, retrieves relevant document chunks, and uses a series of prompts to first create a structured JSON evaluation and then a user-facing formal response.

## ✅ Prerequisites

Before you begin, ensure you have the following installed on your system:

  * **Python 3.8+**: [Installation Guide](https://www.python.org/downloads/)
  * **Docker and Docker Compose**: Required to run the Milvus vector database.
      * [Install Docker Engine](https://docs.docker.com/engine/install/)
      * [Install Docker Compose](https://docs.docker.com/compose/install/)
  * **Ollama**: To run the local Large Language Model. [Download and install from the official Ollama website](https://ollama.com/).

## ⚙️ Setup Instructions

Follow these steps to get the project up and running.

### 1\. Clone the Repository

```bash
git clone <your-repository-url>
cd <your-repository-directory>
```

### 2\. Set Up the Milvus Vector Database

We will use Docker Compose to easily start a Milvus instance.

  * Create a file named `docker-compose.yml` in the root of your project directory.
  * Copy and paste the following content into the file:

<!-- end list -->

```yaml
version: '3.5'

services:
  etcd:
    container_name: milvus-etcd
    image: quay.io/coreos/etcd:v3.5.5
    environment:
      - ETCD_AUTO_COMPACTION_MODE=revision
      - ETCD_AUTO_COMPACTION_RETENTION=1000
      - ETCD_QUOTA_BACKEND_BYTES=4294967296
      - ETCD_HEARTBEAT_INTERVAL=500
      - ETCD_ELECTION_TIMEOUT=2500
    volumes:
      - ${DOCKER_VOLUME_DIRECTORY:-.}/volumes/etcd:/etcd
    command: etcd -advertise-client-urls=http://127.0.0.1:2379 -listen-client-urls http://0.0.0.0:2379 --data-dir /etcd

  minio:
    container_name: milvus-minio
    image: minio/minio:RELEASE.2022-09-07T22-19-44Z
    environment:
      MINIO_ACCESS_KEY: minioadmin
      MINIO_SECRET_KEY: minioadmin
    volumes:
      - ${DOCKER_VOLUME_DIRECTORY:-.}/volumes/minio:/minio_data
    command: minio server /minio_data
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 30s
      timeout: 20s
      retries: 3

  standalone:
    container_name: milvus-standalone
    image: milvusdb/milvus:v2.3.1
    command: ["milvus", "run", "standalone"]
    environment:
      ETCD_ENDPOINTS: etcd:2379
      MINIO_ADDRESS: minio:9000
    volumes:
      - ${DOCKER_VOLUME_DIRECTORY:-.}/volumes/milvus:/var/lib/milvus
    ports:
      - "19530:19530"
      - "9091:9091"
    depends_on:
      - "etcd"
      - "minio"

networks:
  default:
    name: milvus
```

  * Start the Milvus service:

<!-- end list -->

```bash
docker-compose up -d

```

You can check if the containers are running with `docker ps`.

### 3\. Install Python Dependencies

Install all the required Python packages using pip:

```bash
pip install langchain langchain-community langchain-core langchain-huggingface langchain-milvus langchain_ollama pymilvus pypdf docx2txt
```

### 4\. Download a Local LLM with Ollama

The script is configured to use the `llama3` model. Pull it from the Ollama library by running the following command in your terminal:

```bash
ollama pull llama3
```

Ensure the Ollama application is running in the background.

## 🏃‍♀️ Running the System

Follow this two-step process to run the claim evaluator.

### Step 1: Ingest Your Documents

First, you need to populate the vector database with your insurance policy documents.

1.  Create a folder named `source_documents` in your project's root directory.

2.  Place your insurance policy files (PDFs, DOCX, etc.) inside the `source_documents` folder.

3.  Run the ingestion script:

    ```bash
    python ingest.py
    ```

    You will see progress as it loads, splits, and embeds the documents into Milvus. This only needs to be done once for a given set of documents.

### Step 2: Run a Claim Query

Once your documents are ingested, you can process a claim.

1.  Open the `query_systemCV.py` file and inspect the `sample_query` variable if you wish to change it.

    ```python
    # Inside query_systemCV.py
    sample_query = "My son, who is 19, needs dental braces. Our family policy is 2 years old. Is this covered?"
    ```

2.  Run the query script from your terminal:

    ```bash
    python query_systemCV.py
    ```

### Example Output

The script will first print the internal, structured JSON decision that the AI made. This is useful for auditing and logging.

```json
--- INTERNAL JSON RESULT ---
{
  "decision": "Rejected",
  "amount": 0,
  "justification": "The claim for dental braces for the 19-year-old son is rejected. The policy's Orthodontic Care clause explicitly states that coverage for braces is limited to dependents under the age of 18 at the time of treatment initiation. As the son is 19, this condition is not met.",
  "referenced_clauses": [
    {
      "source": "source_documents/Family_Health_Policy.pdf",
      "page": 3
    },
    {
      "source": "source_documents/Dental_Rider_v2.pdf",
      "page": 1
    }
  ]
}
```

Finally, it will print the clean, formal response intended for the policyholder.

```
--- FORMAL RESPONSE FOR USER ---
Dear Policyholder,

We have reviewed your claim regarding coverage for dental braces.

Based on the terms of your policy, the claim has been rejected. The justification for this decision is that the policy's Orthodontic Care clause specifies that coverage for braces is limited to dependents who are under the age of 18 when the treatment begins.

Should you have further questions, please refer to your policy documents.
```

## 🧹 Clean Up

To stop and remove the Milvus containers created by Docker Compose, run:

```bash
docker-compose down
```Of course, here is a README file for your GitHub repository.

-----

# AI-Powered Insurance Claim Evaluator

This project demonstrates a sophisticated, multi-stage AI system for evaluating insurance claims. It uses a Retrieval-Augmented Generation (RAG) architecture to provide decisions based on specific policy documents. The system first ingests and indexes insurance policy documents into a vector database. Then, it processes a user's claim, retrieves relevant policy clauses, and makes a structured, auditable decision, which is finally translated into a formal response for the policyholder.

## 🚀 Features

  * **Document Ingestion**: Loads and processes PDF or DOCX documents from a source folder.
  * **Vectorization**: Splits documents into chunks and creates embeddings using `HuggingFace` models.
  * **Vector Storage**: Stores and indexes document vectors using **Milvus** for efficient similarity search.
  * **Multi-Stage Processing**:
    1.  **Query Parsing**: Understands the user's plain-language query and extracts key information into a structured JSON format.
    2.  **Contextual Retrieval**: Fetches the most relevant policy clauses from Milvus based on the parsed query.
    3.  **AI-Powered Decision Making**: Uses a Large Language Model (LLM) via **Ollama** to evaluate the claim against the retrieved clauses and generate a structured JSON decision.
    4.  **Formal Response Generation**: Translates the internal JSON decision into a formal, professional response for the end-user.
  * **Local & Open-Source**: Runs entirely on your local machine using powerful open-source tools like Ollama, Milvus, and HuggingFace.

## 🛠️ System Architecture

The project consists of two main Python scripts:

1.  **`ingest.py`**: This script is responsible for the one-time setup of the knowledge base. It scans a designated folder (`source_documents/`), loads all the policy files, splits them into manageable chunks, generates vector embeddings for each chunk, and stores them in a Milvus vector database collection. This process only needs to be run once, or whenever the source policy documents are updated.

2.  **`query_systemCV.py`**: This is the main application logic. It takes a user's query (e.g., "Is my son's dental surgery covered?") and orchestrates the RAG pipeline. It initializes the connection to Milvus and the Ollama LLM, retrieves relevant document chunks, and uses a series of prompts to first create a structured JSON evaluation and then a user-facing formal response.

## ✅ Prerequisites

Before you begin, ensure you have the following installed on your system:

  * **Python 3.8+**: [Installation Guide](https://www.python.org/downloads/)
  * **Docker and Docker Compose**: Required to run the Milvus vector database.
      * [Install Docker Engine](https://docs.docker.com/engine/install/)
      * [Install Docker Compose](https://docs.docker.com/compose/install/)
  * **Ollama**: To run the local Large Language Model. [Download and install from the official Ollama website](https://ollama.com/).

## ⚙️ Setup Instructions

Follow these steps to get the project up and running.

### 1\. Clone the Repository

```bash
git clone <your-repository-url>
cd <your-repository-directory>
```

### 2\. Set Up the Milvus Vector Database

We will use Docker Compose to easily start a Milvus instance.

  * Create a file named `docker-compose.yml` in the root of your project directory.
  * Copy and paste the following content into the file:

<!-- end list -->

```yaml
version: '3.5'

services:
  etcd:
    container_name: milvus-etcd
    image: quay.io/coreos/etcd:v3.5.5
    environment:
      - ETCD_AUTO_COMPACTION_MODE=revision
      - ETCD_AUTO_COMPACTION_RETENTION=1000
      - ETCD_QUOTA_BACKEND_BYTES=4294967296
      - ETCD_HEARTBEAT_INTERVAL=500
      - ETCD_ELECTION_TIMEOUT=2500
    volumes:
      - ${DOCKER_VOLUME_DIRECTORY:-.}/volumes/etcd:/etcd
    command: etcd -advertise-client-urls=http://127.0.0.1:2379 -listen-client-urls http://0.0.0.0:2379 --data-dir /etcd

  minio:
    container_name: milvus-minio
    image: minio/minio:RELEASE.2022-09-07T22-19-44Z
    environment:
      MINIO_ACCESS_KEY: minioadmin
      MINIO_SECRET_KEY: minioadmin
    volumes:
      - ${DOCKER_VOLUME_DIRECTORY:-.}/volumes/minio:/minio_data
    command: minio server /minio_data
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 30s
      timeout: 20s
      retries: 3

  standalone:
    container_name: milvus-standalone
    image: milvusdb/milvus:v2.3.1
    command: ["milvus", "run", "standalone"]
    environment:
      ETCD_ENDPOINTS: etcd:2379
      MINIO_ADDRESS: minio:9000
    volumes:
      - ${DOCKER_VOLUME_DIRECTORY:-.}/volumes/milvus:/var/lib/milvus
    ports:
      - "19530:19530"
      - "9091:9091"
    depends_on:
      - "etcd"
      - "minio"

networks:
  default:
    name: milvus
```

  * Start the Milvus service:

<!-- end list -->

```bash
docker-compose up -d
```

You can check if the containers are running with `docker ps`.

### 3\. Install Python Dependencies

Install all the required Python packages using pip:

```bash
pip install langchain langchain-community langchain-core langchain-huggingface langchain-milvus langchain_ollama pymilvus pypdf docx2txt
```

### 4\. Download a Local LLM with Ollama

The script is configured to use the `llama3` model. Pull it from the Ollama library by running the following command in your terminal:

```bash
ollama pull llama3
```

Ensure the Ollama application is running in the background.

## 🏃‍♀️ Running the System

Follow this two-step process to run the claim evaluator.

### Step 1: Ingest Your Documents

First, you need to populate the vector database with your insurance policy documents.

1.  Create a folder named `source_documents` in your project's root directory.

2.  Place your insurance policy files (PDFs, DOCX, etc.) inside the `source_documents` folder.

3.  Run the ingestion script:

    ```bash
    python ingest.py
    ```

    You will see progress as it loads, splits, and embeds the documents into Milvus. This only needs to be done once for a given set of documents.

### Step 2: Run a Claim Query

Once your documents are ingested, you can process a claim.

1.  Open the `query_systemCV.py` file and inspect the `sample_query` variable if you wish to change it.

    ```python
    # Inside query_systemCV.py
    sample_query = "My son, who is 19, needs dental braces. Our family policy is 2 years old. Is this covered?"
    ```

2.  Run the query script from your terminal:

    ```bash
    python query_systemCV.py
    ```

### Example Output

The script will first print the internal, structured JSON decision that the AI made. This is useful for auditing and logging.

```json
--- INTERNAL JSON RESULT ---
{
  "decision": "Rejected",
  "amount": 0,
  "justification": "The claim for dental braces for the 19-year-old son is rejected. The policy's Orthodontic Care clause explicitly states that coverage for braces is limited to dependents under the age of 18 at the time of treatment initiation. As the son is 19, this condition is not met.",
  "referenced_clauses": [
    {
      "source": "source_documents/Family_Health_Policy.pdf",
      "page": 3
    },
    {
      "source": "source_documents/Dental_Rider_v2.pdf",
      "page": 1
    }
  ]
}
```

Finally, it will print the clean, formal response intended for the policyholder.

```
--- FORMAL RESPONSE FOR USER ---
Dear Policyholder,

We have reviewed your claim regarding coverage for dental braces.

Based on the terms of your policy, the claim has been rejected. The justification for this decision is that the policy's Orthodontic Care clause specifies that coverage for braces is limited to dependents who are under the age of 18 when the treatment begins.

Should you have further questions, please refer to your policy documents.
```

## 🧹 Clean Up

To stop and remove the Milvus containers created by Docker Compose, run:

```bash
docker-compose down
```
