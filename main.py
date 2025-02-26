from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_pinecone import PineconeVectorStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
import os
from pinecone import Pinecone, ServerlessSpec, Index # Import Index

# Load environment variables from .env file
load_dotenv(find_dotenv())

# Check if the API key is set
if "OPENAI_API_KEY" not in os.environ:
    raise ValueError(
        "OPENAI_API_KEY environment variable not set. "
        "Please set it in a .env file or as an environment variable."
    )
if "PINECONE_API_KEY" not in os.environ or "PINECONE_ENV" not in os.environ:
    raise ValueError(
        "PINECONE_API_KEY or PINECONE_ENV environment variable not set. "
        "Please set it in a .env file or as an environment variable."
    )

try:
    # Initialize the ChatOpenAI LLM and Embeddings model
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
    embeddings = OpenAIEmbeddings()

    # Initialize Pinecone
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index_name = "langchain-chatbot"

    # Check if the index exists (Corrected)
    existing_indexes = [index.name for index in pc.list_indexes()]  # get the names from the index
    if index_name not in existing_indexes:
        print(f"Creating index '{index_name}'...")
        pc.create_index(
            index_name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=os.environ["PINECONE_ENV"])  # use serverless spec
        )

    # Initialize the vector store (Corrected)
    vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings, namespace="text")
    retriever = vectorstore.as_retriever(search_kwargs=dict(k=3))

    # Define the prompt template
    system_prompt = """You are a helpful AI assistant. 
    You answer the user's questions as best as you can.
    You can use the following context to answer the questions:
    {context}"""

    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_prompt),
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("{input}"),
    ])

    # Define the chain
    def format_docs(docs):  # function that make the documents readable
        return "\n\n".join([d.page_content for d in docs])


    print(retriever.invoke("hello"))  # test the retriever

    chain = (
        {
            "context": lambda x: format_docs(retriever.invoke(x["input"])),  # get the documents and format them.
            "history": lambda x: x["history"],
            "input": lambda x: x["input"]
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    # Initialize the conversation history
    conversation_history = []
    # Initialize the context to an empty list of strings
    initial_context = ""

    while True:
        # Get user input
        user_input = input("You: ")

        if user_input.lower() in ["quit", "exit", "bye"]:
            break

        # Add the user message to vectorstore
        vectorstore.add_texts([user_input])

        # Execute the chain with the conversation history, user input, and retrieved documents
        response = chain.invoke(
            {"history": conversation_history, "input": user_input, "context": initial_context})  # the context is now initialized

        # Add the user message to the conversation history
        conversation_history.append(HumanMessage(content=user_input))
        conversation_history.append(AIMessage(content=response))

        # Print the AI's response content
        print(f"AI: {response}")

except ImportError as e:
    print(f"Error: Missing library. {e}")
    print("Please run 'pip install langchain-openai python-dotenv pinecone langchain-pinecone'")
except ValueError as e:
    print(f"Error: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
