# Friends Script Generator

# Overview
The Friends Script Generator is a deep learning project that utilizes GPT-2 to generate scripts in the style of the popular TV show Friends. This project aims to showcase the capabilities of generative text models in creating humorous and contextually relevant dialogues resembling the iconic sitcom.

## Why Friends?
*In general, Friends is a good choice of a show to use for this project:*

*   Limited settings
    * Monica's apartment
    * Chandler and Joey's apartment
    * Central Perk
    * Ross's apartment
*   Limited characters for an ensemble cast show (6 main characters with distinct personalities and humors)
*   All scripts from all seasons are avaiable
*   Consistent title naming ("The One With ... ")
*   Notable vernacular/ slang
    * Joey's catchphrase, "How you doin'?"
    * The characters use the emphasized word "so" to modify adjectives more often than any other intensifier
    * Chandler's habit of ending a sentence unfinished for sarcasm
* Friends has also been credited in helping non-English speaking students to learn the language, so it should be a good choice for large language model

Overall, with these consistencies it should be easier for the model to "learn" how the show works, and make it recognizable for the reader.

## Data Acquisition
The dataset consists of scripts from the Friends TV show, webscraped from Crazy For Friends, a Friends fansite. 

## Data Preparation
### Data Exploration
First, explore the dataset to understand its structure and the distribution of dialogues and characters.

### Tokenization
The script data is then tokenized using GPT-2's tokenizer, with texts over the token limit being removed to ensure compatibility with the model's input requirements.

## Training the Model
### Dataset Setup
Prepare the dataset for training by splitting it into training and validation sets.

### DataLoader
Implement a DataLoader for efficient loading and batching of the data during the training process.

### Training Loop
Train the model using the prepared dataset, monitoring for loss and adjusting parameters as necessary.

## Generating Scripts
With the trained model, you can generate new Friends-style scripts by providing seed text or letting the model generate content freely. The generation process is customizable, allowing for adjustments in creativity and coherence.

## Streamlit Demo
A demonstration of how the script generator works, please see app.py

## Critical Analysis
- GPT-2 was not a great model to fine-tune for this
	- We tried to use Llama-2 but kept running into overhead issues
- Friends is a physical comedy, and gibberish scripts do fit for a lot of the way Friends episodes work, but a lot of the comedy is not found on the page so is very hard to imitate
- Creative writing generation is an incredibly difficult task for large language models (and people!) and while our model does give us fun "creative" things, they don't always make a ton of sense

## Model Card
[Friends Script Generator](https://github.com/rachelkmont/friends-script-generator/blob/main/model_card.md)

## Resources

Here are the clickable resource links for a .md readme:

- Episodes are found on: [Crazy For Friends, a Friends fansite.](https://www.livesinabox.com/friends/scripts.shtml)
- [Better Language Models and Their Implications](https://openai.com/research/better-language-models)
- [Wordcraft: Story Writing With Large Language Models](https://dl.acm.org/doi/pdf/10.1145/3490099.3511105)
- [Multi-Task Instruction Tuning of LLaMA for Specific Scenarios:
A Preliminary Study on Writing Assistance](https://arxiv.org/pdf/2305.13225.pdf)
- [LLMs as Writing Assistants: Exploring Perspectives on Sense of
Ownership and Reasoning](https://arxiv.org/pdf/2404.00027.pdf)
- [ProSwitch: Knowledge-Guided Instruction Tuning to Generate
Professional and Non-Professional Styled Text](https://arxiv.org/pdf/2403.09131.pdf)
- [Fine-Tuning Large Language Models: A Step-by-Step Tutorial](https://www.datacamp.com/tutorial/fine-tuning-large-language-models)
- [How to generate text: using different decoding methods for language generation with Transformers](https://github.com/huggingface/blog/blob/main/notebooks/02_how_to_generate.ipynb)
- [Fine-Tuning ChatGPT for Text Generation With W&B](https://wandb.ai/mostafaibrahim17/ml-articles/reports/Fine-Tuning-ChatGPT-for-Text-Generation-With-W-B--Vmlldzo1NDE5MjYw)
- [TV Script Generation using RNNs](https://medium.com/@matthew1992/tv-script-generation-563ba7b6356a)
- [Friends Wiki](https://friends.fandom.com/wiki/Friends)


## Acknowledgements
This project is inspired by the original work of the creators of Friends and the open-source community contributing to the development of the GPT-2 model. 

*Disclaimer for scripts: This project is in no way associated with Friends, Warner Bros, NBC or Bright/Kauffman/Crane Productions. This project is for educational purposes only.*

