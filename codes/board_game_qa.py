import os

import chromadb
import uuid 

from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_ollama import OllamaLLM

from chromadb.utils import embedding_functions
from PyPDF2 import PdfReader

##### -------------------------------------------------------------------------- #####

DATA_PATH = "../data/board_game_rules/"
CHROMA_PATH = "../data/chroma/"

##### -------------------------------------------------------------------------- #####
## --- Create a vector store of the documents with all the fields necessary -- ###

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def chunk_text(text, chunk_size=1000, overlap=200):
    """Splits text into overlapping chunks for embedding."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def get_chroma_collection(chroma_path=CHROMA_PATH, collection_name="board_game_rules"):
    """Returns a Chroma collection, creating it if it does not exist."""
    client = chromadb.PersistentClient(path=chroma_path)
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_func
    )
    return collection

# TODO: add opportunity to recreate chromadb 
def populate_collection_from_pdfs(collection, data_path=DATA_PATH):
    """Populates the Chroma collection with text chunks from PDFs."""
    if len(collection.get()["ids"]) > 0:
        return  # Already populated

    for file in os.listdir(data_path):
        if file.endswith(".pdf"):
            path = os.path.join(data_path, file)
            text = extract_text_from_pdf(path)
            chunks = chunk_text(text)

            collection.add(
                ids=[str(uuid.uuid4()) for _ in chunks],
                documents=chunks,
                metadatas=[{"source": file, "line": line} for line in range(len(chunks))]
            )

##### -------------------------------------------------------------------------- #####
## --- A retriever-ranker layer to identify relevant context --- ##

def retrieve_relevant_docs(collection, question, n_results=5):
    """Retrieves top-k relevant documents for a question using embedding similarity and cross-encoder ranking."""
    retriever_model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")
    query_embedding = retriever_model.encode(question, convert_to_numpy=True)

    query_results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=n_results
    )

    retrieved_docs = query_results["documents"][0]
    retrieved_sources = query_results["metadatas"][0]

    # Rank documents using CrossEncoder
    ranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    candidates = [(question, doc) for doc in retrieved_docs]
    relevance_scores = ranker_model.predict(candidates)

    ranked_results = sorted(
        zip(relevance_scores, retrieved_docs, retrieved_sources),
        key=lambda x: x[0],
        reverse=True
    )
    
    _, results, sources = zip(*ranked_results)
    return results, sources

##### -------------------------------------------------------------------------- #####
## --- A pretrained LLM layer to generate the final answer --- ##
def answer_question_with_llm(question, context_docs, sources, model_name="llama3.2"):
    """Uses an LLM to answer a question based on context documents."""
    model = OllamaLLM(model=model_name)
    response = model.invoke(f"""
You are a helpful assistant.
Use the following context to answer the question clearly and completely in some coherent sentences.
If you don't know the know the answer, just write 'Sorry, I cannot help to answer the question based on my dataset.'

The ranked relevant context is the following in a list:
{context_docs}

The question:
{question}

Your answer should end with a short note indicating the source(s) of the information.
For example: "Source: something.pdf"

The available source files for this context are:
{sources}

Your answer:
""")
    return response


def main(question):
    collection = get_chroma_collection()
    populate_collection_from_pdfs(collection)
    context_docs, sources = retrieve_relevant_docs(collection, question)
    answer = answer_question_with_llm(question, context_docs, sources)
    return answer

if __name__ == "__main__":
    user_input_question = "What types of wound exist in the Dead of Winter?" 
    #user_input_question = "What are the main differences between Dead of Winter and 7 Wonders?"
    response = main(user_input_question)
    print("Model response:")
    print(response)


### Streamlit interface ###