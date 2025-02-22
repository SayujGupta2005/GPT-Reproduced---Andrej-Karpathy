# GPT Language Model (Reproduction of Andrej Karpathy's Code)
# Author : Sayuj Gupta
# Date : 13.2.2025

This repository contains a reproduction of Andrej Karpathy's GPT implementation. The code is recreated for **learning and practice purposes**, and the comments reflect my understanding from the original code and various other sources. This project is not an original implementation but rather a way to deepen my knowledge of Transformer-based language models.

## Features
- 'bigram.py' is a the foundation of our transformer model which basically exists to develop how a context is achieved from previous characters.
- 'transformer.py' implements a **Transformer-based GPT model** from scratch using PyTorch.
- Includes **multi-head self-attention** and **feedforward layers**.
- Uses **token and positional embeddings**.
- Performs **causal self-attention** for text generation.
- Supports **training on custom datasets**.

---

## Setup and Installation
### **1. Install Dependencies**
This project requires Python and PyTorch. Install the necessary packages:
```sh
pip install torch numpy tqdm
```
If you want to run it on a GPU, install PyTorch with CUDA:
```sh
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### **2. Run the Model**
To test the model, you can execute:
```sh
python transformer.py
```
This will start training the Transformer model on a toy dataset that is 'input.txt'(Shakespeare Dataset) in our case.
---

## **Project Structure**
```
📂 gpt-reproduction
├── 📜 README.md          # This file
├── 📜 bigram.py           # Basic character generator using bigram model
├── 📜 transformer.py      # Transformer implementation
|__ 📜 input.txt         # Dataset preparation
```

---

## **Transformer Model Overview**
The model consists of the following key components:

### **1. Token & Positional Embeddings**
- Converts input token indices into dense **word vectors**.
- Adds **positional encodings** to retain word order.

### **2. Transformer Blocks**
- Each block contains:
  - **Multi-head self-attention**: Enables context-aware token interactions.
  - **Feedforward network**: Non-linear transformation.
  - **Layer normalization**: Stabilizes training.

### **3. Final Output Layer**
- Maps the **Transformer output** to vocabulary logits.
- Uses **CrossEntropy loss** for training.
---

## **Usage**
### **1. Modify Hyperparameters**
Change `n_embd`, `n_head`, `n_layer`, etc., in `transformer.py` to adjust model size.

### **2. Train on Custom Data**
- Replace the dataset in `input.txt` with your own text corpus.
- For training, comment everything after line 229 and run the transformer.py
- This will start the training and save the model after every defined intervals

### **3. Save and Load Model**
- For eval, comment everything from line 206 to 229 and uncomment after that
- This will prevent training loop from running and jump directly to evaluation part
- Model will be loaded from defined path (update path as per your need)
- The input text is basically some random tensor generated consisting of zeros which doesn't matter.

---
## **Note**
- Even if gpu is enabled, the model will take a lot of time to train so be ready for it
- Model couldn't be uploaded because of large size
## **References**
- [Andrej Karpathy's Original Code](https://github.com/karpathy/ng-video-lecture)
- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)

---

## **License**
This project is **not an original implementation** but a learning exercise. Feel free to use and modify it for educational purposes.

