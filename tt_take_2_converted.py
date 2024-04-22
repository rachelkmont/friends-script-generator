import os
import glob
import re
import tensorflow as tf
import numpy as np
from collections import Counter
import torch
import pickle
from transformers import pipeline, AdamW, GPT2Tokenizer, GPT2LMHeadModel
from tqdm.auto import tqdm
import torch.nn as nn  # For neural network modules
import torch.optim as optim  # For optimizers like SGD, Adam, etc.
import torch.nn.functional as F  # For functions like activations
from torch.utils.data import DataLoader, Dataset  # For creating data loaders and custom datasets
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from shutil import copyfile

from google.colab import drive
drive.mount('/content/drive')

directory_path = '/content/drive/My Drive/Colab'
directory_files = os.listdir(directory_path)

# Define the path to the directory
directory = "/content/drive/My Drive/Colab/friends/cleaned_scripts"


# ## Tokenize Data

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

original_directory = "/content/drive/My Drive/Colab/friends/split_scripts"
new_directory = "/content/drive/My Drive/Colab/friends/gpt_scripts"

if not os.path.exists(new_directory):
    os.makedirs(new_directory)

# Initialize tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Process files
for filename in os.listdir(original_directory):
    if filename.endswith('.txt'):
        filepath = os.path.join(original_directory, filename)
        with open(filepath, 'r', encoding='utf-8') as file:
            script_content = file.read()

        # Tokenize and check length
        tokens = tokenizer.encode(script_content)
        if len(tokens) <= 1024:  # GPT-2 token limit
            # Copy file to new directory
            new_filepath = os.path.join(new_directory, filename)
            copyfile(filepath, new_filepath)

print("Finished processing scripts.")


# ## Set Up the Training Dataset

# Ensure your custom dataset class can handle a list of tokenized scripts:

def collate_fn(batch):
    batch_padded = pad_sequence(batch, batch_first=True, padding_value=0)
    return batch_padded

class ScriptDataset(Dataset):
    def __init__(self, directory):
        self.tokenized_scripts = []
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

        # Read and tokenize each script in the directory
        for filename in os.listdir(directory):
            if filename.endswith('.txt'):
                filepath = os.path.join(directory, filename)
                with open(filepath, 'r', encoding='utf-8') as file:
                    script_content = file.read()
                    tokens = tokenizer.encode(script_content, truncation=True, max_length=1024)
                    self.tokenized_scripts.append(torch.tensor(tokens, dtype=torch.long))

    def __len__(self):
        return len(self.tokenized_scripts)

    def __getitem__(self, idx):
        return self.tokenized_scripts[idx]

# Usage
new_directory = "/content/drive/My Drive/Colab/friends/gpt_scripts"

# Create new directory if it doesn't exist
if not os.path.exists(new_directory):
    os.makedirs(new_directory)

dataset = ScriptDataset(new_directory)


# Create Dataset and DataLoader
batch_size = 8
dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

for batch in dataloader:
    print(type(batch))
    if isinstance(batch, dict):
        print(batch.keys())
    # Add more debug prints here if needed
    break  # Remove or comment out this line to inspect more batches


# ## Prepare Data for Training

model = GPT2LMHeadModel.from_pretrained("gpt2")
model.train()


# ### Training loop

# Load tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

def train_model(model, dataloader, epochs=3, lr=5e-5, max_seq_length=512):
    optimizer = AdamW(model.parameters(), lr=lr)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        epoch_loss = 0
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}/{epochs}")

        for i, batch in progress_bar:
            # Check if the batch is a tensor
            if not isinstance(batch, torch.Tensor):
                raise ValueError("Each batch should be a tensor.")

            # Assume the batch is [input_ids, labels]
            # Adjust this if your batch structure is different
            input_ids = batch[:, :max_seq_length].to(device)
            labels = batch[:, :max_seq_length].to(device)  # Adjust this line based on your specific label structure

            optimizer.zero_grad()

            try:
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                progress_bar.set_postfix({'loss': loss.item()})
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print(f"WARNING: out of memory with batch {i}. If this message repeats, reduce batch size.")
                    torch.cuda.empty_cache()
                else:
                    raise e

        print(f"Average Loss Epoch {epoch+1}: {epoch_loss / len(dataloader)}")


train_model(model, dataloader, epochs=3)


# ## Save the trained model

# Define the paths where you want to save the model and tokenizer
model_save_path = '/content/drive/My Drive/Colab/friends/gpt2_model'
tokenizer_save_path = '/content/drive/My Drive/Colab/friends/gpt2_model'

# Save the model and tokenizer
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(tokenizer_save_path)

## load the trained model and tokenizer
model = GPT2LMHeadModel.from_pretrained(model_save_path)
tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_save_path)

# Make sure to set the model to the correct device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# ## Generate TV Script
# This will generate the TV script for you.  Set `gen_length` to the length of TV script you want to generate.

scene = "Monica and Rachel's Apartment"
gen_length = 2
temperature = 0.8  # Adjust as needed
top_k = 40         # Adjust as needed
top_p = 0.9        # Adjust as needed
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def generate_script_from_prompt(model, tokenizer, prompt, gen_length, device, temperature=0.7, top_k=50, top_p=0.95):
    model.eval()
    with torch.no_grad():
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

        output = model.generate(
            input_ids,
            max_length=gen_length,
            do_sample=True,  # Enable sampling-based generation
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            no_repeat_ngram_size=2,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

        generated_script = tokenizer.decode(output[0], skip_special_tokens=True)

        # Post-process to remove sequences of repetitive punctuation
        cleaned_script = re.sub(r'[\!\?\.\,]{2,}', '', generated_script)

    return cleaned_script

# Example usage
scene_prompt = "Central Perk. Rachel and Ross discuss their plans for the evening."
gen_length = 150
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

generated_script = generate_script_from_prompt(model, tokenizer, scene_prompt, gen_length, device)
print(generated_script)

