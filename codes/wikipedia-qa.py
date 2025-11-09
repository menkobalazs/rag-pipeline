import os

import wikipedia
import wikipediaapi 

import chromadb
import uuid # for unique ids

from sentence_transformers import SentenceTransformer, util, CrossEncoder
from langchain_ollama import OllamaLLM



##### -------------------------------------------------------------------------- #####
### Get the user input question ###

user_input_question = "What is KPMG, and what does it focus on? When does it sponsor Lenovo?" 
#user_input_question = "In what ways did the Industrial Revolution transform everyday life in 19th-century Europe?" 


##### -------------------------------------------------------------------------- #####
### Find key phrases to search on Wikipedia ###

model = OllamaLLM(model="llama3.2")

question_keywords = model.invoke(f"""
You are a clever and helpful assistant that extracts key Wikipedia-searchable keyword structures from user questions.
Given a question or questions, output 2-4 keyword phrases (each no more than 4 words) that could be used as Wikipedia search queries to answer it.

Rules:
- Do not rephrase the question as a full sentence.
- Focus on core topics, entities, or events.
- Include time periods, places, and main subjects when relevant.
- Avoid filler words ('in what ways,' 'how did,' 'why').
- Output as a list of keyword structures separated like '; ' and nothing else                           
                                 
The question(s):
{user_input_question}

Your answer:
"""
).split('; ')

##### -------------------------------------------------------------------------- #####
### Search some Wikipedia articles for the question keywords ###

### search wikipedia pages ###
search_results = []
for query_kw in question_keywords:
    search_results.append(wikipedia.search(query=query_kw, results=1, suggestion=False)[0])
    
### create folder to store Wikipedia articles
data_folder_path = f"../data/wiki_datasets/{question_keywords[0].replace(' ', '_')}/"
os.makedirs(data_folder_path, exist_ok=True) 

### TODO: Save urls and show after the answers as source. ###
### save wikipedia articles ###
wiki_texts = []
wiki_text_lengths: list[int] = []
for i, search_item in enumerate(search_results):
    page = wikipediaapi.Wikipedia(user_agent='user@nowhere.com', language='en').page(search_item)
    page_text = page.text.replace('\n\n', '\n')
    wiki_texts.append(page_text.splitlines())
    wiki_text_lengths.append(len(page_text.splitlines()))
    # link = page.fullurl # save it 
    with open(data_folder_path+f'{i+1:02d}_'+search_item.replace(' ', '_')+'.txt', "w") as wikipage:
        wikipage.write(page_text)
        
        
print(f">>> Wikipedia pages are saved to '{data_folder_path}'.")


##### -------------------------------------------------------------------------- #####
## --- Create a vector store of the documents with all the fields necessary -- ###
print(">>> Create chromadb.")
client = chromadb.PersistentClient(path=data_folder_path)

print(">>> Create collection.")
collection = client.get_or_create_collection(name='wiki_pages')

print(">>> Add data to collection.")
for i, k in enumerate(wiki_texts):
    collection.add(
        ids=[str(uuid.uuid4()) for _ in k],
        documents=k,
        metadatas=[{"page" : i, "line": line} for line in range(len(k))]
    )
    print(f">>>>> {i+1}/{len(wiki_texts)}. item added.")
print(">>> Collection created.")


##### -------------------------------------------------------------------------- #####
## --- A retriever-ranker layer to identify relevant context --- ##

retriever_model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")

# Encode query and documents
documents = [line for doc in wiki_texts for line in doc if line.strip()]
doc_embs = retriever_model.encode(documents, convert_to_tensor=True)
query_emb = retriever_model.encode(user_input_question, convert_to_tensor=True)

# Retrieve top-k candidates
K = 5
scores = util.cos_sim(query_emb, doc_embs)[0]
top_results = scores.topk(k=K)

candidates = [(user_input_question, documents[idx]) for idx in top_results.indices]

ranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
relevance_scores = ranker_model.predict(candidates)
ranked_results = sorted(zip(relevance_scores, candidates), reverse=True)
results = [text for _, (_, text) in ranked_results]


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

Your answer:
"""
)

print("Model response:")
print(response)


### Streamlit interface ###



