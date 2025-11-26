import os
from dotenv import load_dotenv

from langchain.agents import AgentExecutor
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage
from langchain.agents import create_tool_calling_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool

# load environment variables
load_dotenv()

#   INITIALIZE EMBEDDINGS MODEL

embeddings = OllamaEmbeddings(
    model=os.getenv("EMBEDDING_MODEL"),
)

#  INITIALIZE CHROMA VECTOR STORE

vector_store = Chroma(
    collection_name=os.getenv("COLLECTION_NAME"),
    embedding_function=embeddings,
    persist_directory=os.getenv("DATABASE_LOCATION"),
)

#   INITIALIZE CHAT MODEL

llm = init_chat_model(
    os.getenv("CHAT_MODEL"),
    model_provider=os.getenv("MODEL_PROVIDER"),
    temperature=0
)

# Pulling prompt from hub

prompt = PromptTemplate.from_template("""                                
You are a helpful assistant. You will be provided with a query and a chat history.
Your task is to retrieve relevant information from the vector store and provide a response.
For this you use the tool 'retrieve' to get the relevant information.

The query is as follows:                    
{input}

The chat history is as follows:
{chat_history}

Please provide a concise and informative response based on the retrieved information.
You can retrieve information only once.
If you don't know the answer, say "I don't know" (and don't provide a source).

You can use the scratchpad to store any intermediate results or notes.
The scratchpad is as follows:
{agent_scratchpad}

For every piece of information you provide, also provide the source.

Return text as follows:

<Answer to the question>
Source: source_url
""")


# Creating the retriever tool
@tool
def retrieve(query: str):
    """Retrieve information related to a query."""
    retrieved_docs = vector_store.similarity_search(query, k=1)

    serialized = ""

    for doc in retrieved_docs:
        serialized += f"Source: {doc.metadata['source']}\nContent: {doc.page_content}\n\n"

    return serialized


# Combining all tools
tools = [retrieve]

# Initiating the agent
agent = create_tool_calling_agent(llm, tools, prompt)

# Create the agent executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Initialize chat history
messages = []

print("🤖 Agentic RAG Chatbot")
print("Type 'quit' to exit the chat")

while True:
    # Get user input
    user_question = input("\nYou: ")

    if user_question.lower() == 'bye':
        break

    # Add user message to history
    messages.append(HumanMessage(content=user_question))

    # Invoke the agent
    result = agent_executor.invoke({"input": user_question, "chat_history": messages})

    ai_message = result["output"]

    # Add AI message to history
    messages.append(AIMessage(content=ai_message))
