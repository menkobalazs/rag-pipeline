import os
import wikipedia
import wikipediaapi 
import chromadb
import uuid 
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_ollama import OllamaLLM


##### -------------------------------------------------------------------------- #####

DATA_PATH = "../data/wiki_datasets/"

##### -------------------------------------------------------------------------- #####
### Find key phrases to search on Wikipedia ###

def extract_keywords_from_question(question, model_name="llama3.2", max_keywords=4):
    """Extract 2-4 keyword phrases from a question using an LLM."""
    model = OllamaLLM(model=model_name)
    keywords = model.invoke(f"""
You are a clever and helpful assistant that extracts key Wikipedia-searchable keyword structures from user questions.
Given a question or questions, output 2-4 keyword phrases (each no more than 4 words) that could be used as Wikipedia search queries to answer it.

Rules:
- Do not rephrase the question as a full sentence.
- Focus on core topics, entities, or events.
- Include time periods, places, and main subjects when relevant.
- Avoid filler words ('in what ways,' 'how did,' 'why').
- Output as a list of keyword structures separated like '; ' and nothing else                           

The question(s):
{question}

Your answer:
""").split('; ')
    
    return keywords[:max_keywords]

##### -------------------------------------------------------------------------- #####
### Search some Wikipedia articles for the question keywords ###

# TODO: Save urls and show after the answers as source. #
def fetch_wikipedia_articles(keywords, data_folder_root=DATA_PATH):
    """Searches Wikipedia and saves articles to a folder. Returns article texts and file paths."""
    search_results = [wikipedia.search(query=kw, results=1, suggestion=False)[0] for kw in keywords]

    data_folder_path = os.path.join(data_folder_root, keywords[0].replace(' ', '_'))
    os.makedirs(data_folder_path, exist_ok=True)

    wiki_texts = []
    wiki_file_paths = []
    wiki_api = wikipediaapi.Wikipedia(user_agent='user@nowhere.com', language='en')

    for i, search_item in enumerate(search_results):
        page = wiki_api.page(search_item)
        page_text = page.text.replace('\n\n', '\n')
        # link = page.fullurl 
        wiki_texts.append(page_text.splitlines())

        file_path = os.path.join(data_folder_path, f'{i+1:02d}_{search_item.replace(" ", "_")}.txt')
        wiki_file_paths.append(file_path)
        
        with open(file_path, "w") as wikipage:
            wikipage.write(page_text)
    
    print(f">>> Wikipedia pages saved to '{data_folder_path}'.")
    return wiki_texts, wiki_file_paths, data_folder_path


##### -------------------------------------------------------------------------- #####
## --- Create a vector store of the documents with all the fields necessary -- ###

def create_chroma_collection(data_folder_path, wiki_texts):
    """Create a Chroma collection from Wikipedia article lines."""
    client = chromadb.PersistentClient(path=data_folder_path)
    collection = client.get_or_create_collection(name='wiki_pages')

    for i, doc_lines in enumerate(wiki_texts):
        collection.add(
            ids=[str(uuid.uuid4()) for _ in doc_lines],
            documents=doc_lines,
            metadatas=[{"page": i, "line": line} for line in range(len(doc_lines))]
        )
        print(f">>>>> {i+1}/{len(wiki_texts)}. item added.")
    
    print(">>> Collection created.")
    return collection

##### -------------------------------------------------------------------------- #####
## --- A retriever-ranker layer to identify relevant context --- ##

def retrieve_relevant_docs(collection, wiki_texts, question, top_k=5):
    """Retrieve and rank the most relevant document lines using embeddings + cross-encoder."""
    retriever_model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")
    documents = [line for doc in wiki_texts for line in doc if line.strip()]
    embeddings = retriever_model.encode(documents, convert_to_numpy=True)

    query_embedding = retriever_model.encode(question, convert_to_numpy=True)

    query_results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k
    )

    retrieved_docs = query_results['documents'][0]

    candidates = [(question, doc) for doc in retrieved_docs]
    ranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    relevance_scores = ranker_model.predict(candidates)

    ranked_results = sorted(zip(relevance_scores, retrieved_docs), reverse=True)
    results = [doc for _, doc in ranked_results]
    return results

##### -------------------------------------------------------------------------- #####
## --- A pretrained LLM layer to generate the final answer --- ##

def answer_question_with_llm(question, context_docs, model_name="llama3.2"):
    """Generate an answer to a question based on retrieved context."""
    model = OllamaLLM(model=model_name)
    response = model.invoke(f"""
You are a helpful assistant.
Use the following context to answer the question clearly and completely in some coherent sentences.
If you don't know the know the answer, just write 'Sorry, I cannot help to answer the question based on my dataset.'

The ranked relevant context is the following in a list:
{context_docs}

The question:
{question}

Your answer:
""")
    return response

##### -------------------------------------------------------------------------- #####

def main(question):
    keywords = extract_keywords_from_question(question)
    wiki_texts, wiki_file_paths, data_folder_path = fetch_wikipedia_articles(keywords)
    collection = create_chroma_collection(data_folder_path, wiki_texts)
    context_docs = retrieve_relevant_docs(collection, wiki_texts, question)
    answer = answer_question_with_llm(question, context_docs)
    return answer, wiki_file_paths


if __name__ == "__main__":
    user_question = "What is KPMG, and what does it focus on? When does it sponsor Lenovo?"
    response, sources = main(user_question)
    print("Model response:")
    print(response)
    print("\nSources:")
    print(sources)


### Streamlit interface ###
