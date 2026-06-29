# Conditional-date-generator-generative-ai

A deep learning project that generates valid calendar dates from structured input conditions using multiple conditional generative neural network architectures. The project was developed as part of the **DSAI 490 – Assignment 2** course and investigates different sequence generation approaches for structured data.

---

## Overview

This project formulates date generation as a **conditional sequence-to-sequence generation** task.

Given a set of conditions including:

* Day of the week
* Month
* Leap year flag
* Decade

the model generates a valid date satisfying all provided constraints.

To study different generative learning techniques, five neural architectures were implemented and compared.

---

## Implemented Models

* Transformer
* Seq2Seq LSTM
* Conditional Autoencoder (CAE)
* Conditional Variational Autoencoder (CVAE)
* Conditional Generative Adversarial Network (CGAN)

---

## Features

* Custom character-level tokenizer
* Conditional sequence generation
* Transformer-based architecture
* Seq2Seq encoder-decoder implementation
* Conditional Autoencoder and Variational Autoencoder
* Conditional GAN implementation
* Class-weighted training for imbalanced data
* Automatic training, validation, and inference pipelines
* PyTorch implementation

---

## Technologies

* Python
* PyTorch
* Transformer
* LSTM
* GRU
* Variational Autoencoder (VAE)
* Generative Adversarial Network (GAN)
* NumPy

---

## Project Structure

```text
├── train.py
├── predict.py
├── dataset.py
├── models.py
├── weights/
└── data/
```

---

## Results

The Transformer and Seq2Seq LSTM achieved the best generation performance, successfully learning valid date structures from conditional inputs. The Conditional VAE demonstrated the ability to model diverse latent representations, while the Conditional GAN highlighted the challenges of training GANs for discrete sequence generation.

---

## Learning Outcomes

* Sequence-to-Sequence Learning
* Conditional Text Generation
* Transformer Architectures
* Autoencoders & Variational Autoencoders
* Generative Adversarial Networks
* Character-Level Tokenization
* Deep Learning Model Evaluation
* PyTorch Model Development

---

## Author

**Zeyad Sherif**
