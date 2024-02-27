# friends-script-generator

Generating the Friends TV scripts using deep learning

In this project, we'll generate our own Simpsons TV scripts using RNNs. we'll be using scripts of the Simpsons found online. The Neural Network we'll build will generate a new TV script for a scene chosen by the user.

Credits for scripts:

Of course, I'd be happy to help you update the README for your Friends script generator project. I'll start by reviewing the main processing file you've shared to understand how your project works, including how it preprocesses data, generates scripts, and handles deliverables. This will allow me to suggest an updated README structure and content that reflects the purpose and functionality of your project accurately.

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

## Acknowledgements
This project is inspired by the original work of the creators of Friends and the open-source community contributing to the development of the GPT-2 model.

