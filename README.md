# unfake-ai
*A Deep Learning and Crowdsourcing-Based Fake News Detection Framework*

---

## Overview

This repository contains the **core system model and algorithmic logic** developed as part of the *Unfake* project — a social media fake news detection system integrating **Deep Learning**, **Crowdsourcing**, and **Blockchain** technologies.  

This repository specifically represents **the author’s individual contribution**, focusing on:
- The **AI model** (BERT + CNN architecture) for misinformation detection.
- The **Checked Algorithm**, which fuses AI predictions with credibility-weighted user evaluations to produce adaptive, explainable fake news classifications.

Together, these components form the **intelligence layer** of Unfake, enabling accurate, balanced, and transparent verification of online news content.

If you'd like to learn more about Unfake, watch this video:
https://youtu.be/hF0d_B1q3Yo?si=QpRMxKmn4rnpK3bd

---

## Why This Project Is Useful

The rapid spread of fake news on social media platforms like *X* (formerly Twitter) has made traditional detection methods insufficient. Purely AI-based systems often suffer from contextual bias, while human-only approaches lack scalability and consistency.  

This project introduces a **hybrid AI–human framework** that addresses these gaps:

- **Hybrid Decision Logic** – Combines model predictions and user votes using dynamic entropy-based weighting.  
- **Deep Learning Accuracy** – Employs a fine-tuned BERT + CNN model for text classification with high contextual understanding.  
- **Adaptive Reliability** – Adjusts influence between AI and crowd input based on confidence levels and agreement entropy.  
- **Transparency and Fairness** – Prevents bias, trolling, or overreliance on uncertain model outputs.  

This solution enhances **credibility assessment, interpretability, and trust** — all essential for real-world misinformation detection.

---

## What You'll Find

### 1. `Model/`
Contains all components related to the **BERT + CNN deep learning model** for fake news detection.

- `[1] model_run.py` – Python script to execute and evaluate the model.  
- `[2] model_train.py` – Python script to train and tune the model.  
- `[3] model_overview.md` – Documentation detailing:
  - The model framework and processing pipeline  
  - Dataset description
  - Training configuration and hyperparameters used  

### 2. `Algorithm/`
Contains all components related to the **Checked Algorithm**, which integrates AI predictions and crowdsourced evaluations.

- `[1] checked_algorithm.py` – Python script to execute the Checked Algorithm.  
- `[2] algorithm_logic.md` – Documentation explaining:
  - The logic behind the Checked Algorithm  
  - The weighting mechanism between AI and user inputs  
  - Entropy-based confidence and reliability calculations  

Each folder includes all files necessary to understand, reproduce, or extend the author’s individual contribution to the Unfake system.

---

## Getting Help

If you encounter issues, wish to extend this work, or require clarification:

- Open an issue in this repository’s **[Issues](../../issues)** section.

---

## Maintainer & Contributor

**Author / Maintainer:**  
**Han Wei Chang**  
- Bachelor of Computer Science (Hons), Taylor’s University  
- Roles: AI Model Developer & Algorithm Engineer  

**Key Contributions:**
- Designed and implemented the **BERT + CNN fake news detection model**.  
- Developed the **Checked Algorithm** for hybrid AI–crowdsourcing decision-making.  
- Integrated entropy-based weighting for adaptive reliability.

This repository represents **only the author’s individual component** of the *Unfake* project.

---

## Citation

If you reference this work in academic or technical research, please cite it as:

> Chang, H. W. (2025). *Unfake: Social Media Fake News Detection System via Crowdsourcing, Deep Learning, and Blockchain Technologies.* Taylor’s University School of Computer Science.
