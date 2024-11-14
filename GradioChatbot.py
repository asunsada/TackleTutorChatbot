#### This script creates a chatbot application hosted in HuggingFace Spaces
# using a fine-tuned model by me. the finetuning is done using Kaggle notebook:
#https://www.kaggle.com/code/asunsada/reg2-q-a-fine-tune-gemma-models-in-keras-using-lor


import os
import gradio as gr
import keras_hub
import os

import re




#gemma_llm = keras_hub.models.GemmaCausalLM.from_preset(model_path)
#print(gemma_llm)

model_path="kaggle://asunsada/gemma2_2b_it_en_roleplay/keras/football_coach_11042024_epoch15" # kaggle


class GemmaChatbot:
    def __init__(self):
        # Initialize the model
        #preset = "gemma_instruct_2b_en"  # name of pretrained Gemma 2
        #self.gemma_llm = keras_hub.models.GemmaCausalLM.from_preset(preset)
        # Load your custom model
        self.gemma_llm = keras_hub.models.GemmaCausalLM.from_preset(model_path)
        print(self.gemma_llm)

    def format_prompt(self, message, history):
        # Format conversation history into a single string
        formatted_history = ""
        for user_msg, assistant_msg in history:
            formatted_history += f"User: {user_msg}\nAssistant: {assistant_msg}\n"

        # Add the current message
        prompt = formatted_history + f"User: {message}\nAssistant:"
        return prompt

    def generate_response(self, message, history):
        # Format the prompt with history

        #prompt = self.format_prompt(message, history)
        prompt= message
        # Generate response
        output = self.gemma_llm.generate(prompt,256)
        output = clean_incomplete_sentences(output)  # remove  incomplete sentences (typically
        # last one or any question at the end.
        return output.replace(prompt, "").strip()

# remove  incomplete sentences (typically last one or any question at the end.)
def clean_incomplete_sentences(text):
    # Split text into sentences using regular expressions
    sentences = re.split(r'(?<=\.) |(?<=\?) |(?<=!) ', text)

    # Filter out incomplete sentences (those not ending with ".", "?" or "!")
    complete_sentences = [s for s in sentences if re.search(r'[.!?]$', s)]

    # Remove the last sentence if it ends with a question mark
    if complete_sentences and complete_sentences[-1].endswith('?'):
        complete_sentences = complete_sentences[:-1]

    # Join sentences back into a single string
    cleaned_text = ' '.join(complete_sentences)
    return cleaned_text

def create_chatbot():
    # Initialize the chatbot
    chatbot = GemmaChatbot()

    # Create the Gradio interface
    chat_interface = gr.ChatInterface(
        fn=chatbot.generate_response,
        title="🏈 Tackle Tutor Chatbot 🏈 💬",
        description="I'm Tackle Tutor, the head coach of the greatest football team around! "
            "With over 20 years of coaching experience and numerous championships under my belt, "
            "I've also had the honor of coaching and playing in the NFL. \n\n"
            "Ask me how to tackle any challenge, on the field or in life, and I'll guide you through it",
        examples=[
            "Why is resilience important in football and life?",
            "How can I keep trying when something is really hard?",
            "How do famous coaches inspire?",
            "What can I learn from coach Barry Switzer?",
            "How can football teach us about teamwork?",
            "Why is preparation essential for success?",
            "Why should we always give our best effort?",
        ],
        theme=gr.themes.Soft()
    )

    return chat_interface


# Launch the chatbot
if __name__ == "__main__":
    # Create and launch the interface
    chat_interface = create_chatbot()
    chat_interface.launch(share=True)