# ai-paper-explorer/main.py
import os
import arxiv
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.chains.summarize import load_summarize_chain
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType

load_dotenv()

# Constants
PAPERS_DB = "./vectorstore/papers_db"
SAVED_DB = "./vectorstore/saved_db"

# Setup
llm = ChatOpenAI(temperature=0.3)
embeddings = OpenAIEmbeddings()
papers_store = Chroma(persist_directory=PAPERS_DB, embedding_function=embeddings)
saved_store = Chroma(persist_directory=SAVED_DB, embedding_function=embeddings)

# Summarize a paper
def summarize_text(text):
    chain = load_summarize_chain(llm, chain_type="stuff")
    return chain.run([Document(page_content=text)]).strip()

# Fetch and summarize new papers from arXiv
def fetch_and_summarize_papers(query="artificial intelligence", max_results=10):
    print("Fetching latest AI papers from arXiv...")
    search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.SubmittedDate)
    new_papers = []

    for result in search.results():
        if papers_store.similarity_search(result.title, k=1):
            continue

        summary = summarize_text(result.summary)
        doc = Document(
            page_content=result.summary,
            metadata={
                "title": result.title,
                "authors": [a.name for a in result.authors],
                "url": result.entry_id,
                "summary": summary
            }
        )
        papers_store.add_documents([doc])
        new_papers.append(doc)

    papers_store.persist()
    print(f"Added {len(new_papers)} new papers.")

# Format output for a document
def format_metadata(doc):
    meta = doc.metadata
    return f"Title: {meta.get('title')}\nAuthors: {', '.join(meta.get('authors', []))}\nSummary: {meta.get('summary')}\nURL: {meta.get('url')}\n"

# Retrieve QA chain
def get_qa_chain(store):
    return RetrievalQA.from_chain_type(llm=llm, retriever=store.as_retriever(), chain_type="stuff")

# Add/remove paper to/from saved DB
def add_to_saved(title):
    docs = papers_store.similarity_search(title, k=1)
    if docs:
        saved_store.add_documents(docs)
        saved_store.persist()
        return f"Added '{title}' to saved papers."
    return "Paper not found in current papers."

def delete_from_saved(title):
    docs = saved_store.similarity_search(title, k=1)
    if docs:
        ids = [doc.metadata.get("title") for doc in docs]
        saved_store._collection.delete(where={"title": {"$in": ids}})
        saved_store.persist()
        return f"Deleted '{title}' from saved papers."
    return "Paper not found in saved papers."

def add_or_remove_saved(cmd):
    if "add" in cmd.lower():
        return add_to_saved(cmd)
    elif "delete" in cmd.lower():
        return delete_from_saved(cmd)
    else:
        return "Please specify 'add' or 'delete' in your command."

# Launch CLI agent
def launch_agent():
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    tools = [
        Tool(name="SearchPapers", func=lambda q: get_qa_chain(papers_store).run(q), description="Search recent AI papers."),
        Tool(name="SavedPapers", func=lambda q: get_qa_chain(saved_store).run(q), description="Search your saved AI papers."),
        Tool(name="ModifySaved", func=lambda q: add_or_remove_saved(q), description="Add or remove papers from saved DB with 'add' or 'delete' in query.")
    ]

    agent = initialize_agent(tools=tools, llm=llm, agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, memory=memory, verbose=False)

    print("\n[Agent Ready] Ask about AI research papers or type 'exit' to quit.\n")
    while True:
        query = input("You: ")
        if query.lower() in ["exit", "quit"]:
            print("Exiting. Goodbye!")
            break
        try:
            response = agent.run(query)
            print(f"Agent: {response}\n")
        except Exception as e:
            print(f"Error: {e}\n")

if __name__ == "__main__":
    fetch_and_summarize_papers()
    launch_agent()