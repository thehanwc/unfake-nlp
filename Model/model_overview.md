# Model Overview  
**BERT + CNN Fake News Detection Model**  
*Component of the Unfake Project – Author: Han Wei Chang*

---

## 1. Framework Overview

The **BERT + CNN model** serves as the deep learning foundation of *Unfake*, designed to classify social media posts as *Real* or *Fake*.  
This hybrid architecture combines the **contextual understanding of BERT (Bidirectional Encoder Representations from Transformers)** with the **local feature extraction power of Convolutional Neural Networks (CNNs)**, optimizing performance on short-text, high-variance data typical of social media platforms.

### Objectives
- Accurately detect misinformation across diverse text formats.  
- Preserve semantic context while enhancing feature discriminability.  
- Serve as the primary AI component in the Checked Algorithm’s decision pipeline.

---

## 2. Infrastructure & Environment

| Component | Description |
|------------|-------------|
| **Language** | Python 3.10 |
| **Frameworks** | PyTorch, Transformers (HuggingFace) |
| **Hardware** | NVIDIA GPU (RTX 3060 or higher recommended) |
| **Environment Tools** | CUDA 12.0, cuDNN, Optuna for hyperparameter tuning |
| **Operating System** | Windows 11 / Ubuntu 22.04 LTS |

All experiments were executed in a reproducible virtual environment with dependency tracking to ensure consistency across runs.

---

## 3. Model Architecture

### Base Architecture
The architecture integrates **BERT (base-uncased)** as an encoder, followed by lightweight CNN layers for localized pattern learning:

Input Text
↓
BERT Tokenizer → BERT Encoder
↓
CNN (1D convolutional layers with multiple kernel sizes)
↓
Global Max Pooling
↓
Fully Connected Layers (ReLU + Dropout)
↓
Softmax Output (2 classes: Real, Fake)

### Design Rationale
- **BERT** captures contextual dependencies and word semantics.
- **CNN layers** emphasize short-span phrase patterns (n-grams) important for detecting fake-news linguistic cues.
- **Dropout** mitigates overfitting.
- **Softmax** provides normalized probability outputs for downstream algorithm integration.

---

## 4. Data and Dataset Used

| Aspect | Description |
|---------|-------------|
| **Dataset Source** | WELFake Dataset (2021) |
| **Content Type** | Social media text posts and online news headlines/articles |
| **Language** | English only |
| **Classes** | 2 (Real, Fake) |
| **Dataset Split** | 70% Training, 15% Validation, 15% Testing |
| **Preprocessing** | Tokenization via BERT tokenizer, lowercasing, punctuation removal, maximum sequence length of 128 tokens |

The dataset was cleaned to remove duplicates, short (<10 words) entries, and non-informative samples.

---

## 5. Model Development Pipeline

1. **Data Preparation**  
   - Data loaded and labeled into “real” and “fake”.  
   - Text tokenized using BERT’s WordPiece tokenizer.

2. **Feature Extraction**  
   - BERT embeddings extracted per token and fed into CNN.  
   - Three parallel convolution layers (kernel sizes 2, 3, 4) capture varying phrase lengths.

3. **Model Training**  
   - Optimizer: `AdamW`  
   - Loss Function: `CrossEntropyLoss`  
   - Regularization: `Dropout = 0.2`  
   - Batch Size: 32  
   - Learning Rate: 2e-5  
   - Epochs: 5–10 (early stopping applied)

4. **Evaluation**  
   - Metrics: Accuracy, Precision, Recall, F1-score  
   - Model checkpoint saved as `model_weights.pth`

5. **Hyperparameter Tuning**  
   - Automated tuning via **Optuna** for learning rate, dropout, and batch size.  
   - Best-performing configuration used for final model export.

---

## 6. Key Parameters

| Parameter | Value | Description |
|------------|--------|-------------|
| `model_name` | `bert-base-uncased` | Pretrained BERT encoder |
| `max_length` | 128 | Max token length per sequence |
| `learning_rate` | 2e-5 | Optimizer learning rate |
| `dropout` | 0.2 | Dropout to prevent overfitting |
| `batch_size` | 32 | Number of samples per gradient update |
| `epochs` | 8 | Total training iterations |
| `optimizer` | AdamW | Weight decay variant of Adam |
| `loss_function` | CrossEntropyLoss | Binary classification loss |
| `kernel_sizes` | [2, 3, 4] | CNN filter window sizes |
| `filters_per_layer` | 100 | Number of CNN feature maps per kernel |

---

## 7. Integration into Unfake System

The trained model provides the **AI authenticity score (P_AI)** for each piece of content.  
This score is then passed into the **Checked Algorithm**, where it is weighted dynamically against user credibility inputs to produce the final authenticity verdict.

---

**Author:** Han Wei Chang  
**Institution:** Taylor’s University – School of Computer Science   

