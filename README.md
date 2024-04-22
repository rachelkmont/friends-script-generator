# Friends Script Generator

## Project Overview
The Friends Script Generator is a deep learning project that utilizes GPT-2 to generate scripts in the style of the popular TV show Friends. This project aims to showcase the capabilities of generative text models in creating humorous and contextually relevant dialogues resembling the iconic sitcom.

## Environment Setup
To run this project, you will need Python 3.8+ and the following libraries:
- torch
- transformers
- pandas
- numpy

Installation command:
```bash
pip install torch transformers pandas numpy
```

## Data Acquisition
The dataset consists of scripts from the Friends TV show. While the dataset is not included in this repository due to copyright concerns, it can be acquired from publicly available sources or through personal collection of episodes' subtitles.

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

## Deliverables
The project outputs generated scripts that mimic the style and humor of Friends. These scripts can be found in the `outputs` directory after running the generation script.

## Resources
Here are the clickable resource links for a .md readme:

- [Episodes are found on Crazy For Friends, a Friends fansite.](https://www.livesinabox.com/friends/scripts.shtml)
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

