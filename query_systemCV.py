import json
# Use the new, more specific langchain packages
from langchain_ollama.chat_models import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus.vectorstores import Milvus
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# --- Configuration ---
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
COLLECTION_NAME = "insurance_policies_db"
LLM_MODEL = "llama3"  # Use the model name you have downloaded

# --- Component Initialization ---
print("Initializing components...")
# LLM for structured JSON output
json_llm = ChatOllama(model=LLM_MODEL, format="json")
# LLM for natural language conversation/generation
conversational_llm = ChatOllama(model=LLM_MODEL)
# Embeddings model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# Connect to the existing Milvus collection
vector_store = Milvus(
    embedding_function=embeddings,
    collection_name=COLLECTION_NAME,
    connection_args={"host": MILVUS_HOST, "port": MILVUS_PORT},
)
retriever = vector_store.as_retriever(search_kwargs={'k': 5})
print("Components initialized successfully.")

# --- Prompt Definitions ---

# Prompt to parse the user query (no changes)
QUERY_PARSER_PROMPT = PromptTemplate(
    template="""
    You are an expert at parsing user queries into a structured JSON format.
    Extract the key details from the query below. If a detail is not present, use null.
    Query: {query}
    JSON Output:
    """,
    input_variables=["query"],
)

# Prompt for the final decision-making step (no changes)
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

# --- NEW: Prompt for the Formal Response Layer ---
FORMAL_RESPONSE_PROMPT = PromptTemplate(
    template="""
    You are an AI assistant for an insurance company's claims department.
    Your task is to provide a formal reply to a policyholder about their claim decision.
    The decision details are provided below in a JSON object.

    - The reply must be professional and direct.
    - Clearly state the final decision ("Approved" or "Rejected").
    - Use the 'justification' from the JSON to explain the reason for the decision in a formal tone. Do not simply copy it.
    - Do not add any information not present in the provided JSON.

    Decision JSON:
    {decision_json_str}

    Formal Reply:
    """,
    input_variables=["decision_json_str"],
)


# --- Function Definitions ---

def process_claim(query: str):
    """
    This is the backend logic that produces the structured JSON.
    """
    print(f"\nProcessing query: '{query}'")

    # 1. Parse the query
    print("Step 1: Parsing user query...")
    parser_chain = QUERY_PARSER_PROMPT | json_llm | JsonOutputParser()
    parsed_query = parser_chain.invoke({"query": query})
    print(f" -> Parsed Query: {json.dumps(parsed_query)}")

    # 2. Retrieve relevant clauses
    print("Step 2: Retrieving relevant clauses...")
    retrieved_docs = retriever.invoke(query)
    context = "\n---\n".join([doc.page_content for doc in retrieved_docs])
    print(f" -> Retrieved {len(retrieved_docs)} relevant clauses.")

    # 3. Evaluate and generate the final decision
    print("Step 3: Evaluating and making a final decision...")
    decision_chain = DECISION_MAKER_PROMPT | json_llm | JsonOutputParser()
    final_decision = decision_chain.invoke({
        "parsed_query": json.dumps(parsed_query),
        "context": context
    })
    final_decision['referenced_clauses'] = [doc.metadata for doc in retrieved_docs]

    return final_decision


# --- NEW: Function for the Formal Response Layer ---
def generate_formal_response(decision_json: dict):
    """
    Takes the structured JSON decision and converts it into a
    formal letter for the user.
    """
    # This chain uses the formal prompt and the standard LLM
    response_chain = FORMAL_RESPONSE_PROMPT | conversational_llm

    formal_output = response_chain.invoke({
        "decision_json_str": json.dumps(decision_json, indent=2)
    })

    return formal_output


# --- Main Execution Block ---

if __name__ == '__main__':
    try:
        sample_query = "My son, who is 19, needs dental braces. Our family policy is 2 years old. Is this covered?"

        # 1. Get the structured JSON result from the backend system
        structured_result = process_claim(sample_query)

        print("\n--- INTERNAL JSON RESULT ---")
        print(json.dumps(structured_result, indent=2))

        # 2. Pass the JSON to the formal response layer
        print("\n--- FORMAL RESPONSE FOR USER ---")
        formal_response = generate_formal_response(structured_result)

        # Print the clean text content of the response
        print(formal_response.content)

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("\nPlease ensure both the Docker (Milvus) and Ollama services are running.")
