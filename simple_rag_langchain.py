
#CELL-NO: 4
# Standard library imports
import os

from pathlib import Path

# Environment variable management - for secure API key handling
from dotenv import load_dotenv

# LangChain Document Loaders - for loading PDF documents
from langchain_community.document_loaders import PyPDFLoader

# LangChain Text Splitters - for breaking documents into manageable chunks
from langchain_text_splitters import RecursiveCharacterTextSplitter

# OpenAI Integration - for embeddings and LLM
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# Vector Store - Chroma for efficient similarity search
from langchain_chroma import Chroma

# LangChain Core Components
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from typing import List

#CELL-NO: 5
# Load environment variables from .env file
load_dotenv()

# Verify API key is loaded
if not os.getenv("OPENAI_API_KEY"):
    print("⚠️  WARNING: OPENAI_API_KEY not found!")
    raise ValueError("Missing open ai key")
else:
    print("✓ OpenAI API Key loaded successfully!")
    print(f"✓ Key starts with: {os.getenv('OPENAI_API_KEY')[:8]}...")

def get_document_chunks() -> List[Document]:

  #CELL-NO: 7
  # Example: Loading multiple PDFs from a directory
  pdf_directory = "./pdfs"  # Directory containing your PDFs
  all_documents = []

  if os.path.exists(pdf_directory):
      pdf_files = list(Path(pdf_directory).glob("*.pdf"))
      print(f"Found {len(pdf_files)} PDF files")
      
      for pdf_file in pdf_files:
          loader = PyPDFLoader(str(pdf_file)) #convert Path object to string
          docs = loader.load()
          all_documents.extend(docs) #adds a iterable to list
          print(f"  ✓ Loaded {len(docs)} pages from {pdf_file.name}")
      
      print(f"\nTotal pages loaded: {len(all_documents)}")
  else:
    raise ValueError(f"{pdf_directory} does not exists")

  #CELL-NO: 8
  # Initialize the text splitter with recommended settings
  text_splitter = RecursiveCharacterTextSplitter(
      chunk_size=1024,        # Maximum characters per chunk (roughly 200-250 tokens)
      chunk_overlap=128,      # Characters overlap between chunks (maintains context)
      length_function=len,    # Function to measure chunk length
      separators=["\n\n", "\n", " ", ""]  # Try to split on paragraphs first, then lines, etc.
  )

  # Split the documents into chunks
  # This creates smaller, manageable pieces while preserving semantic meaning
  document_chunks = text_splitter.split_documents(all_documents)

  # Display splitting results
  print(f"✓ Split {len(all_documents)} documents into {len(document_chunks)} chunks")
  print(f"\nAverage chunk size: {sum(len(chunk.page_content) for chunk in document_chunks) / len(document_chunks):.0f} characters")

  return document_chunks


#CELL-NO: 9
# Initialize OpenAI Embeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",  # Latest, cost-effective embedding model
    # dimensions=1536
    # Alternative: "text-embedding-3-large" for better quality
)


persist_directory = "./chroma_db"

if os.path.exists(persist_directory):
    print(f"Loading existing ChromaDB from {persist_directory}...")
    vectorstore = Chroma(
      persist_directory=persist_directory,
      collection_name="simple_rag",
      embedding_function=embeddings
    )
else:
    
    document_chunks = get_document_chunks()

    #CELL-NO: 13
    # create ChromaDB vector store
    print(f"Creating ChromaDB from {len(document_chunks)} chunks...")
    vectorstore = Chroma.from_documents(
        documents=document_chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name="simple_rag"
    )

#CELL-NO: 16
# Initialize the ChatOpenAI model
llm = ChatOpenAI(
      model="gpt-4-turbo-2024-04-09",  # Choose your model
      # Alternative options:
      # model="gpt-4o",           # Faster GPT-4 performance, good 
      # balance,
      # model="gpt-3.5-turbo",    # Faster and cheaper option

      temperature=0,         # 0 = deterministic, factual responses (recommended for Q&A)
      max_tokens=2000,       # Maximum length of response
  )

print("✓ LLM configured successfully")


#CELL-NO: 17
#Step 1: Create the Retriever 
# Create a retriever from the vector store
retriever = vectorstore.as_retriever(
      search_type="similarity",    # Use cosine similarity for search
      search_kwargs={"k": 4}        # Retrieve top 4 most relevant chunks
  )

print("✓ Retriever configured successfully")

#CELL-NO: 18
# Define the prompt template for the RAG system
# This tells the LLM how to use the retrieved context
system_prompt = (
    "You are a helpful assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer based on the context, say that you don't know. "
    "Keep the answer concise and accurate.\n\n"
    "Context: {context}\n\n"
    "Question: {question}"
)

# Create the prompt template
prompt = ChatPromptTemplate.from_template(system_prompt)

# Helper function to format documents
def format_docs(docs):
    """Format retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)

# Build the RAG chain using LangChain 1.0+ LCEL (LangChain Expression Language)
# This uses the pipe operator (|) to chain components together
rag_chain = (
    {
        "context": retriever | format_docs,  # Retrieve docs and format them
        "question": RunnablePassthrough()      # Pass through placeholder for the question
    }
    | prompt           # Format with prompt template
    | llm              # Generate answer with LLM
    | StrOutputParser() # Parse output to string
)

print("✓ RAG chain created successfully using LangChain 1.0+ LCEL!")

#CELL-NO: 19
# Example Query 1: General question about the document
# query1 = "What is the main topic or subject of this document?"
query1 = input("Enter the query: ")

print(f"Query: {query1}")
print("\nProcessing...\n")

# With LangChain 1.0+, we invoke the chain with the question directly
answer = rag_chain.invoke(query1)

print("=" * 80)
print("ANSWER:")
print("=" * 80)
print(answer)
print("\n" + "=" * 80)

# To see which documents were retrieved, we can call the retriever separately
print("\nSOURCE DOCUMENTS USED:")
print("=" * 80)
retrieved_docs = retriever.invoke(query1)
for i, doc in enumerate(retrieved_docs):
    print(f"\nDocument {i+1}:")
    print(f"  Source: {doc.metadata}")
    print(f"  Content: {doc.page_content[:200]}...")
    print("-" * 80)
