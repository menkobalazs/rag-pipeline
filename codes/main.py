
import wikipedia
import wikipediaapi 
import chromadb
import uuid # for unique ids
import os

NUM_OF_WIKI_PAGES = 1

##### -------------------------------------------------------------------------- #####
### Get the user input question ###
## >>> TODO: Find a free process/api that can filter keywords from the question <<< ##
## >>> Until the keyword search feature is implemented, I’ll use Wikipedia’s results. <<< ##


#def get_keywords_from_user_question(user_input_question: str) -> list[text]:
#    pass

user_input_question_basic = "What is KPMG, and what does it focus on?" 
user_input_question = "In what ways did the Industrial Revolution transform social structures and everyday life in 19th-century Europe?"
  
#question_key_words = get_keywords_from_user_question(user_input_question=user_input_question)

## >>> TODO: Generate data folder name <<< ##
data_folder_path = "../data/test_dataset/"
os.makedirs(data_folder_path, exist_ok=True) 

##### -------------------------------------------------------------------------- #####
### Search some Wikipedia articles for the question keywords ###
## >>> TODO: decide the number of 'result'  <<< ##


### search wikipedia pages ###
search_results = wikipedia.search(query=user_input_question, results=NUM_OF_WIKI_PAGES, suggestion=False) 
### save wikipedia articles ###
wiki_texts = []
wiki_text_lengths: list[int] = []
for i, search_item in enumerate(search_results):
    wiki_wiki = wikipediaapi.Wikipedia(user_agent='user@nowhere.com', language='en')
    page = wikipediaapi.Wikipedia(user_agent='user@nowhere.com', language='en').page(search_item)
    page_text = page.text.replace('\n\n', '\n')
    wiki_texts.append(page_text.splitlines())
    wiki_text_lengths.append(len(page_text.splitlines()))
    # link = page.fullurl
    with open(data_folder_path+f'{i+1:02d}_'+search_item.replace(' ', '_')+'.txt', "w") as wikipage:
        wikipage.write(page_text)
        
            
print(f">>> Wikipedia pages are saved to '{data_folder_path}'.")
print()
## --- A retriever-ranker layer to identify relevant context -- ###
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
    print(f">>>>> {i+1}. item added.")
print(">>> Collection created.")



## >>> TODO: Find the result section in the wiki pages and cite it. <<< ##

result = collection.query(
    query_texts = [
        user_input_question        
    ],
    n_results=1
)


for i, query_reslts in enumerate(result["documents"]):
    print(f"\nQuery {i}")
    print("\n".join(query_reslts))

## --- A pretrained LLM layer to generate the final answer --- ##




### Streamlit interface ###