# GPT-J Summarizer
## Introduction
This repository presents the development of a summarization model using GPT-J, a large-scale language model developed by EleutherAI, on Google Colab. The objective of the project is to create a model capable of generating concise summaries of given paragraphs. GPT-J was chosen due to its impressive performance in various natural language processing tasks, including text generation and summarization.

### Method of Approach
Data Collection
The dataset for training the summarization model was obtained from the "cnn_dailymail" dataset, which contains news articles and corresponding highlights as summaries.

### Data Preprocessing
The dataset was preprocessed to create prompt tokens that instruct the model to generate summaries. Special completion tokens were added to the prompts to indicate where the generated summary should start.

### Model Selection
GPT-J, a state-of-the-art language model, was chosen as the base model for the summarization task. GPT-J is known for its high-quality text generation capabilities and its ability to handle long-range dependencies.

### Model Fine-tuning
The selected GPT-J model was fine-tuned using the provided reference code with some modifications. The model was adapted to handle the summarization task by adding adapters to the linear and embedding layers. The adapter dimension was set to 2, and it was used to enable the model to generate summaries efficiently.

### Training Process
The model was trained using the Adam8bit optimizer with a learning rate of 1e-4. The batch size was set to 2, and the training process was performed for 3 epochs. The training data was shuffled to enhance the model's generalization.

## References Used
The primary references used for building the summarization model include:

### EleutherAI's GPT-J repository on GitHub: 
The official repository provided details on model architecture, training, and fine-tuning.
### PyTorch Lightning Documentation: 
The PyTorch Lightning library was used to organize the training loop, and its documentation guided the implementation.
### Hugging Face Transformers Documentation: 
The Hugging Face Transformers library provided pre-trained language models and tokenizers that were crucial for this project.
### Weights & Biases (wandb): 
I used the Weights & Biases (wandb) library for logging and tracking the training process.
## Results
The developed summarization model using GPT-J on Google Colab demonstrates impressive results in capturing the essence of the paragraphs. The model is capable of generating accurate and concise summaries, reflecting its understanding of the input context.

## Conclusion
In conclusion, the summarization model using GPT-J on Google Colab has proven to be highly effective in generating accurate and concise summaries for given paragraphs. The fine-tuning process, along with the addition of adapters, has enabled the model to excel in the summarization task. The model's proficiency in understanding context and generating coherent summaries makes it a valuable tool for various natural language processing applications.

Overall, the project has demonstrated successful implementation of GPT-J for the summarization task, providing an essential tool for text summarization and information extraction.

## Limitations
### Hardware Limitations: 
Due to limitations in available hardware (GPU/TPU), the model's generation speed is currently slow, taking around 20 minutes for 30 tokens. This is because the model solely uses the CPU for text generation.
### Memory Usage: 
8-bit fine-tuning and memory optimization reduced memory usage to around 12 GB RAM, but for optimal performance, high-memory GPU setups are recommended.
### Cost Considerations:
Running the model on Colab's free version may incur high GPU usage costs, so consider this when deploying the model.

*Note: Future improvements may include GPU acceleration for faster generation and better accuracy.*
