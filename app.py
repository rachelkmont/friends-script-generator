import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import re

# Load the model and tokenizer
model_path = "gpt2_model"
tokenizer_path = "gpt2_model"
model = GPT2LMHeadModel.from_pretrained(model_path)
tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def generate_script(prompt, gen_length, temperature, top_k, top_p):
    model.eval()
    with torch.no_grad():
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        outputs = model.generate(
            input_ids,
            max_length=gen_length + len(input_ids[0]),
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            no_repeat_ngram_size=2,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text

# Streamlit interface
st.title('The One with the... AI Generated Script')

st.image('friends_header.jpeg', width=700)

with st.sidebar:  # Use the sidebar for settings
    st.header('Configuration')
    user_prompt = st.text_area("Scene Prompt", "Central Perk. Rachel and Ross discuss their plans for the evening.",
                               help="Enter the initial scene or conversation prompt that you want the script to start from.")
    gen_length = st.slider("Length of Generation", 50, 500, 150,
                           help="Total number of words in the generated script. Higher values produce longer scripts.")
    temperature = st.slider("Temperature", 0.5, 1.0, 0.8,
                            help="Controls the randomness. Lower values make the script more predictable.")
    top_k = st.slider("Top K", 0, 100, 40,
                      help="Limits the number of top probability vocab considered at each step of the generation.")
    top_p = st.slider("Top P", 0.0, 1.0, 0.9,
                      help="Focus script generation on the most likely set of words, reducing randomness.")

st.subheader('Generated Script')
if st.sidebar.button('Generate Script'):
    script = generate_script(user_prompt, gen_length, temperature, top_k, top_p)
    st.text_area("", script, height=300)  # Improved placeholder for better aesthetics

st.markdown('### About This App')
st.markdown("""
This app uses the GPT-2 model to generate TV scripts based on your input prompts. Adjust the settings on the left to change how the script is generated.
""")
