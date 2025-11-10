import os

import chromadb
import uuid 

from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_ollama import OllamaLLM

from chromadb.utils import embedding_functions
from PyPDF2 import PdfReader

DATA_PATH = "../data/board_game_rules/"
CHROMA_PATH = "../data/chroma/"

##### -------------------------------------------------------------------------- #####
### Get the user input question ###

#user_input_question = "What types of wound exist in the Dead of Winter?" 
user_input_question = "What are the main differences between Dead of Winter and 7 Wonders?" 

##### -------------------------------------------------------------------------- #####
## --- Create a vector store of the documents with all the fields necessary -- ###

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

collection = chroma_client.get_or_create_collection(
    name="board_game_rules",
    embedding_function=embedding_func
)

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def chunk_text(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


if not len(collection.get()["ids"]):
    for file in os.listdir(DATA_PATH):
        if file.endswith(".pdf"):
            path = os.path.join(DATA_PATH, file)
            text = extract_text_from_pdf(path)
            chunks = chunk_text(text)

            collection.add(
                ids=[str(uuid.uuid4()) for _ in chunks],
                documents=chunks,
                metadatas=[{"source": file, "line": line} for line in range(len(chunks))] 
            )
            


##### -------------------------------------------------------------------------- #####
## --- A retriever-ranker layer to identify relevant context --- ##

retriever_model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")

# Encode the question
query_embedding = retriever_model.encode(user_input_question, convert_to_numpy=True)

# Retrieve top-k candidates from Chroma
K = 5
query_results = collection.query(
    query_embeddings=[query_embedding.tolist()],
    n_results=K
)

retrieved_docs = query_results["documents"][0]
retrieved_sources = query_results["metadatas"][0]
retrieved_scores = query_results["distances"][0]

# Rank them with a CrossEncoder
ranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# Create (question, doc) pairs
candidates = [(user_input_question, doc) for doc in retrieved_docs]

# Predict relevance scores
relevance_scores = ranker_model.predict(candidates)

# Sort results by ranker score
ranked_results = sorted(
    zip(relevance_scores, retrieved_docs, retrieved_sources),
    key=lambda x: x[0],  # sort only by score
    reverse=True
)
_, results, sources = zip(*ranked_results)

##### -------------------------------------------------------------------------- #####
## --- A pretrained LLM layer to generate the final answer --- ##
model = OllamaLLM(model="llama3.2")

response = model.invoke(f"""
You are a helpful assistant.
Use the following context to answer the question clearly and completely in some coherent sentences.
If you don't know the know the answer, just write 'Sorry, I cannot help to answer the question based on my dataset.'

The ranked relevant context is the following in a list:
{results}

The question:
{user_input_question}

Your answer should end with a short note indicating the source(s) of the information.
For example: "Source: something.pdf"

The available source files for this context are:
{sources}

Your answer:
"""
)

print("Model response:")
print(response)



