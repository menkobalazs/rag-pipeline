import streamlit as st
from board_game_qa import main
#from ollama_server import start_server, stop_server  

st.set_page_config(page_title="Board Game Rules Q&A", layout="wide")
st.title("Board Game Rules Question Answering")

question = st.text_area("Enter your question about board games:", height=120)

if st.button("Get Answer"):
    if question.strip():
        with st.spinner("Processing your question..."):
            try:
                #ollama_process = start_server()
                answer = main(question)
                st.subheader("Answer")
                st.write(answer)
            except Exception as e:
                st.error(f"An error occurred: {e}")
            #finally:
            #    stop_server(ollama_process)
    else:
        st.warning("Please enter a question to get an answer.")
        
