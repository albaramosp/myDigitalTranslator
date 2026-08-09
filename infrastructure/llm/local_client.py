"""
Central configuration — edit this file to tune the app.
"""
from langchain_core.prompts import PromptTemplate

from application.ports.llm_client import LlmClient
from domain.llm.llm_request import LlmRequest
from domain.llm.llm_response import LlmResponse

# Paths (relative to project root inside Docker)
DOCS_DIR = "./magma_docs"
DB_DIR = "./magma_db"
MAIN_PROMPT = PromptTemplate.from_template("You are a really cool person.")
# Ollama
OLLAMA_HOST = "http://ollama:11434"  # service name from docker-compose
OLLAMA_MODEL = "llama3.2"            # change to "mistral" or "phi3:mini" etc.
LLM_TEMPERATURE = 0.2

# Embeddings (runs locally, no API needed)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # ~80MB, good quality/speed tradeoff

# RAG settings
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
RETRIEVER_K = 5  # how many chunks to retrieve per query

# UI
APP_TITLE = "Magma"
APP_CAPTION = "Ask me anything about the Bard!"

"""
Builds the RAG chain: retriever + prompt + LLM.
"""
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_chroma import Chroma

"""
Vectorstore
"""
import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

def get_embeddings() -> HuggingFaceEmbeddings:
    """Load local embedding model (downloads once, cached afterwards)."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def build_vectorstore(embeddings: HuggingFaceEmbeddings) -> tuple[Chroma, dict]:
    """
    If a persisted DB exists, load it.
    Otherwise load docs, chunk them, embed and persist.
    Returns the vectorstore and a stats dict for display in the UI.
    """
    if os.path.exists(DB_DIR) and os.listdir(DB_DIR):
        print(f"[vectorstore] Loading existing DB from {DB_DIR}")
        vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
        stats = {
            "status": "loaded",
            "message": "Loaded existing knowledge base from disk.",
            "docs": None,
            "chunks": None,
        }
        return vectorstore, stats

    print(f"[vectorstore] Building DB from docs in {DOCS_DIR}")
    loader = DirectoryLoader(
        DOCS_DIR,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
    )
    docs = loader.load()

    if not docs:
        raise ValueError(
            f"No .txt files found in {DOCS_DIR}. "
            "Please add texts before starting."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)
    print(f"[vectorstore] Created {len(chunks)} chunks from {len(docs)} documents")

    vectorstore = Chroma.from_documents(
        chunks,
        embedding=embeddings,
        persist_directory=DB_DIR,
    )

    filenames = sorted(set(os.path.basename(d.metadata["source"]) for d in docs))
    stats = {
        "status": "built",
        "message": "Knowledge base built from scratch.",
        "docs": filenames,
        "chunks": len(chunks),
    }
    return vectorstore, stats

class LlamaClient(LlmClient):
    def __init__(self):
        try:
            self.chain, self.stats = load_chain()
        except ValueError as e:
            print(e)
            exit(1)

    def generate(self, request: LlmRequest) -> LlmResponse:
        return LlmResponse(self.chain.invoke(request.user_prompt), 0)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def build_chain(vectorstore: Chroma):
    """Assemble the full RAG chain using LCEL."""
    llm = ChatOllama(
        base_url=OLLAMA_HOST,
        model=OLLAMA_MODEL,
        temperature=LLM_TEMPERATURE,
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_K})

    chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | MAIN_PROMPT
            | llm
            | StrOutputParser()
    )

    return chain


def load_chain():
    embeddings = get_embeddings()
    vectorstore, stats = build_vectorstore(embeddings)
    chain = build_chain(vectorstore)
    return chain, stats