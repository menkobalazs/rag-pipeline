# Based on https://github.com/pixegami/rag-tutorial-v2/blob/main/test_rag.py    

from board_game_qa import main as board_game_qa_main

from langchain_ollama import OllamaLLM


def query_and_validate(question, expected_response):
    response_text = board_game_qa_main(question)
    model = OllamaLLM(model="mistral") # optionally use 'mistral' instead of 'llama3.2'   
    prompt = f"""
Expected Response: {expected_response}
Actual Response: {response_text}

(Answer with 'true' or 'false') Does the actual response match the expected response? Don't be strict — accept synonyms as well.
"""
    evaluation_results_str = model.invoke(prompt)
    evaluation_results_str_cleaned = evaluation_results_str.strip().lower()
    print('\n---')
    print(f"Question: {question}")
    print(prompt)

    if "true" in evaluation_results_str_cleaned:
        # Print response in Green if it is correct.
        print(f"Response: {evaluation_results_str_cleaned}")
        return True
    elif "false" in evaluation_results_str_cleaned:
        # Print response in Red if it is incorrect.
        print(f"Response: {evaluation_results_str_cleaned}")
        return False
    else:
        raise ValueError(
            f"Invalid evaluation result. Cannot determine if 'true' or 'false'."
        )
    

def test_dead_of_winter():
    assert query_and_validate(
        question="What types of wound exist in the Dead of Winter?",
        expected_response="There are two types of wounds: a normal wound and a frostbite wound.",
    )
    
def test_sushi_go_party():
    assert query_and_validate(
        question="In Sushi Go Party game what does an egg nigiri worth?",
        expected_response="one point",
    )
    
def test_kill_the_unicorns():
    assert query_and_validate(
        question="How can I instanly win in Kill the Unicorns game?",
        expected_response="Collect double rainbow",
    )
 
# False question
def test_terraforming_mars():
    assert not query_and_validate(
        question="What is the starting temperature in Terraforming Mars game?",
        expected_response="286 Kelvin",
    )
    
# Irrelevant questions
def test_irrelevant_question():
    assert query_and_validate(
        question="When was the first landing on the moon?",
        expected_response="Sorry, I cannot help to answer the question based on my dataset.",
    )