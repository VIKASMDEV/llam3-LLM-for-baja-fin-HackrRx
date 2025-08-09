# query_system.py
import json
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Milvus

# --- Configuration ---
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
COLLECTION_NAME = "insurance_policies_db"

# --- Initialize necessary components ---
llm = Ollama(model="llama3", format="json") # LLM configured for JSON output
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Connect to the existing Milvus collection
vector_store = Milvus(
    embedding_function=embeddings,
    collection_name=COLLECTION_NAME,
    connection_args={"host": MILVUS_HOST, "port": MILVUS_PORT},
)
retriever = vector_store.as_retriever(search_kwargs={'k': 5}) # Retrieve top 5 relevant chunks

# --- Define Prompts for each step ---

# Prompt to parse the user query into a structured format
QUERY_PARSER_PROMPT = PromptTemplate(
    template="""
    You are an expert at parsing user queries into a structured JSON format.
    Extract the key details from the query below. If a detail is not present, use null.
    Query: {query}
    JSON Output:
    """,
    input_variables=["query"],
)

# Prompt for the final decision-making step
DECISION_MAKER_PROMPT = PromptTemplate(
    template="""
    You are an AI insurance claim evaluator. Your task is to provide a decision based ONLY on the provided policy clauses.
    Do not use any external knowledge.

    **User's Claim Details:**
    {parsed_query}

    **Relevant Policy Clauses:**
    {context}

    **Instructions:**
    1. Analyze the User's Claim Details against the provided Policy Clauses.
    2. Determine a final decision: "Approved" or "Rejected".
    3. Specify the payout amount if applicable based on the clauses, otherwise use 0.
    4. Provide a clear justification for your decision, referencing the specific clauses that support it.
    5. Return your final answer as a single, valid JSON object with the keys "decision", "amount", and "justification".
    """,
    input_variables=["parsed_query", "context"],
)

def process_claim(query: str):
    """
    Processes a natural language query to generate a structured decision.
    """
    print(f"Processing query: '{query}'")

    # 1. Parse the query using the LLM
    print("Step 1: Parsing user query...")
    parser_chain = QUERY_PARSER_PROMPT | llm | JsonOutputParser()
    parsed_query = parser_chain.invoke({"query": query})
    print(f" -> Parsed Query: {json.dumps(parsed_query)}")

    # 2. Retrieve relevant clauses from Milvus
    print("Step 2: Retrieving relevant clauses...")
    retrieved_docs = retriever.invoke(query)
    context = "\n---\n".join([doc.page_content for doc in retrieved_docs])
    print(f" -> Retrieved {len(retrieved_docs)} relevant clauses.")

    # 3. Evaluate and generate the final decision
    print("Step 3: Evaluating and making a final decision...")
    decision_chain = DECISION_MAKER_PROMPT | llm | JsonOutputParser()
    final_decision = decision_chain.invoke({
        "parsed_query": json.dumps(parsed_query),
        "context": context
    })

    # Add references for auditability
    final_decision['referenced_clauses'] = [doc.metadata for doc in retrieved_docs]

    return final_decision

if __name__ == '__main__':
    # sample_query = "46-year-old male, knee surgery in Pune, 3-month-old insurance policy"
    sample_query = "My son, who is 19, needs dental braces. Our family policy is 2 years old. Is this covered?"

    result = process_claim(sample_query)

    print("\n--- FINAL DECISION ---")
    print(json.dumps(result, indent=2))