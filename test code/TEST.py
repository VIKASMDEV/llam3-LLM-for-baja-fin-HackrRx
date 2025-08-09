# TEST.py

# 1. This is the correct, modern import path and class name
from langchain_ollama.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

print("Imports successful. Initializing model...")

try:
    # 2. Initialize the ChatOllama model
    # Make sure Ollama is running with 'ollama run llama3'
    llm = ChatOllama(model="llama3")

    # 3. Create a prompt template. Chat models use a list of messages.
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a witty assistant who loves to tell jokes."),
        ("user", "Tell me a joke about {topic}")
    ])

    # 4. Use a simple string output parser
    parser = StrOutputParser()

    # 5. Build the chain
    chain = prompt | llm | parser

    print("Chain built successfully. Invoking...")

    # 6. Invoke the chain and print the result
    response = chain.invoke({"topic": "the internet"})
    print("\nResponse:")
    print(response)

except Exception as e:
    print(f"\nAn error occurred: {e}")
    print("\nPlease ensure the Ollama application is running and the 'llama3' model is available.")