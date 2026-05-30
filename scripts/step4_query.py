import os
import sys
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# --- Step 1: Load the vector store (faiss_db) built in step3 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.abspath(os.path.join(current_dir, "..", "faiss_db"))

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
vectorstore = FAISS.load_local(db_dir, embeddings, allow_dangerous_deserialization=True)

# --- Step 2: Get a question from the user ---
question = input("\nAsk a question about your documents: ")

# --- Step 3: Search for the most relevant chunks ---
# similarity_search returns the top-k document chunks ranked by vector distance
retrieved_docs = vectorstore.similarity_search(question, k=4)

print(f"\n--- Retrieved {len(retrieved_docs)} chunks ---")
for i, doc in enumerate(retrieved_docs):
    source = doc.metadata.get("source", "unknown")
    page = doc.metadata.get("page", "?")
    print(f"\n[Chunk {i+1}] source: {os.path.basename(source)}, page: {page}")
    print(doc.page_content[:200] + "...")

# --- Step 4: Build a prompt with the retrieved chunks as context ---
# The system message tells the LLM how to behave.
# The context (retrieved chunks) is injected between the system message and the question.
context_text = "\n\n---\n\n".join(doc.page_content for doc in retrieved_docs)

prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a helpful research assistant. Answer the user's question "
        "using ONLY the provided document excerpts below. "
        "If the excerpts do not contain the answer, say so honestly. "
        "Cite which document or page your answer comes from when possible."
    )),
    ("user", "Document excerpts:\n\n{context}"),
    ("user", "Question: {question}"),
])

# --- Step 5: Send to Gemini and print the answer ---
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

print("\n--- Generating answer ---\n")

# LangChain chains the prompt + model call
chain = prompt | llm
response = chain.invoke({"context": context_text, "question": question})

print(response.content)
