# Q&A Pipeline using Retrieval-Augmented Generation ([RAG](https://en.wikipedia.org/wiki/Retrieval-augmented_generation))
## Developer: Balázs Menkó
## Target Role: [Data Scientist / ML Engineer](https://kpmg.hrfelho.hu/allasajanlat/1245/data-scientist-machine-learning-engineer) @ [KPMG](https://kpmg.com/xx/en.html)

---

### Overview

This project implements a Question & Answer (Q&A) pipeline based on the RAG approach. The goal is to combine information retrieval and language generation to deliver accurate, context-aware answers from a knowledge base built from Wikipedia articles.

---

## Description

The project is implemented in Python 3.12. It uses:

- **Vector store:** [Chroma](https://www.trychroma.com/)  
- **Pretrained LLM:** [Llama3.2 (3B)](https://ollama.com/library/llama3.2:3b) from [Ollama](https://ollama.com/)  
- **Interface:** [Streamlit](https://streamlit.io/)  
- **Containerization:** [Docker](https://www.docker.com/)  

The pipeline follows these main steps:

1. Extract keywords from the user’s question using `llama3.2` (`extract_keywords_from_question()` function).  
2. Retrieve relevant Wikipedia articles using `wikipedia` and `wikipedia_api` packages (`fetch_wikipedia_articles()` function).  
3. Store document embeddings in Chroma vector database (`create_chroma_collection()` function).  
4. Perform semantic search using [`multi-qa-MiniLM-L6-cos-v1`](https://huggingface.co/sentence-transformers/multi-qa-MiniLM-L6-cos-v1) from `sentence_transformers`.  
5. Generate the final answer with `llama3.2` using the top 5 most relevant contexts.  

---

## Projects

### Project #1 – Wikipedia Q&A Pipeline

The pipeline extracts searchable keywords from user questions, fetches Wikipedia articles, and builds a Chroma vector store. `llama3.2` is used to generate context-aware answers based on the top relevant passages.

---

### Project #2 – Board Game Manuals Q&A

To evaluate the pipeline on a known dataset, 11 board game manuals were used:

- [Terraforming Mars](https://cdn.1j1ju.com/medias/13/3f/fb-terraforming-mars-rule.pdf)  
- [Dead of Winter: A Crossroads Game](https://cdn.1j1ju.com/medias/b8/42/26-dead-of-winter-a-crossroads-game-rulebook.pdf)  
- [Bang!](https://cdn.1j1ju.com/medias/8d/fc/eb-bang-rulebook.pdf)  
- [Kill the Unicorns](https://cdn.1j1ju.com/medias/b9/3e/e1-kill-the-unicorns-rulebook.pdf)  
- [Terra Mystica](https://cdn.1j1ju.com/medias/9c/2c/c8-terra-mystica-rulebook.pdf)  
- [Dune: Imperium](https://cdn.1j1ju.com/medias/17/a9/6d-dune-imperium-rulebook.pdf)  
- [We Didn’t Playtest This at All!](https://asmadigames.com/rules/Playtest_Rules.pdf)  
- [Sushi Go Party!](https://cdn.1j1ju.com/medias/f2/97/92-sushi-go-party-rulebook.pdf)  
- [Azul](https://cdn.1j1ju.com/medias/03/14/fd-azul-rulebook.pdf)  
- [UNO](https://service.mattel.com/instruction_sheets/42001pr.pdf)  
- [Catan](https://www.catan.com/sites/default/files/2021-06/catan_base_rules_2020_200707.pdf)  

The pipeline follows the same steps as the Wikipedia project. A [Streamlit interface](https://github.com/menkobalazs/rag-pipeline/blob/main/codes/streamlit_board_game_qa.py) was created for user-friendly interaction.

A testing mechanism was implemented using the [Mistral 7B](https://ollama.com/library/mistral:7b) model to evaluate whether the generated answers match the expected results. Results are recorded in [`pytest_result.txt`](https://github.com/menkobalazs/rag-pipeline/blob/main/documents/pytest_result.txt).


#### Results
Based on the PyTest, four questions were answered correctly out of five. The test took 15 minutes and 2 seconds, which means an average query can be solved in about three minutes. This seems a little long, but the test was run on a CPU-only machine (Intel Core i7 - 2.4 GHz), which may explain the slowness. 

Fortunately, the failure was not a hallucination. The RAG pipeline correctly identified that the question was not answerable based on the dataset, even though it could theoretically be answered.

---

## References and Sources

**Websites:**

- [HuggingFace Models](https://huggingface.co/models)  
- [RAG Tutorial GitHub](https://github.com/pixegami/rag-tutorial-v2/blob/main/query_data.py)  
- [Ollama Search Model](https://ollama.com/search)  

**YouTube Tutorials:**

- [How to containerize Python applications with Docker](https://youtu.be/0UG2x2iWerk?si=189TKC1ftDOlDUWN)  
- [The Only Docker Tutorial You Need To Get Started](https://youtu.be/DQdB7wFEygo?si=HRQof8xR8gjclYL6)  
- [Chroma Python Basics](https://youtu.be/yvsmkx-Jaj0?si=Kr5wKSnozAVx4ZQB)  
- [Build a RAG in 10 minutes! | Python, ChromaDB, OpenAI](https://youtu.be/JfSmffOyV-8?si=0HmHTLf846t4A1Yb)  
- [Python RAG Tutorial (with Local LLMs): AI For Your PDFs](https://youtu.be/2TJxpyO3ei4?si=GgMX9SFtDpUkcP3x)  

Additionally, ChatGPT was used to improve efficiency during development.
