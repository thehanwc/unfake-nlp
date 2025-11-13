# Algorithm Logic  
**Checked Algorithm**  
*Component of the Unfake Project – Author: Han Wei Chang*

---

## 1. Overview

The **Checked Algorithm** is the decision-making component of *Unfake*, responsible for fusing two sources of information to determine whether a piece of content is **real**, **fake**, or **uncertain**:

1. The **AI model’s authenticity prediction**, expressed as a probability that the content is real.  
2. The **crowdsourced user evaluation**, where each user contributes a vote and a credibility score.

By combining both, the algorithm delivers a **weighted authenticity score** adjusted by **entropy-based confidence**, ensuring fairness between AI precision and human consensus.

---

## 2. Algorithm Objective

The algorithm aims to:
- Balance **AI confidence** and **crowd reliability** through adaptive weighting.  
- Account for **uncertainty (entropy)** in both AI and user inputs.  
- Produce a **final normalized score** in the range [−1, +1].  
- Output a clear and interpretable classification: `"Real"`, `"Fake"`, or `"Not Sure"`.

---

## 3. Parameters

### Constant
| Parameter | Type | Description |
|------------|------|-------------|
| `w_ai_base` | float | Base weighting factor for AI (default = 0.5). |
| `w_user_base` | float | Base weighting factor for users (default = 0.5). |
| `min_votes` | int | Minimum number of user votes required for inclusion (default = 1). |
| `threshold_real` | float | Score above which classification = *Real* (default = +0.34). |
| `threshold_fake` | float | Score below which classification = *Fake* (default = −0.34). |

### Inputs
| Parameter | Type | Description |
|------------|------|-------------|
| `probability_real` | float ∈ [0, 1] | The AI model’s probability estimate that the content is real. |
| `model_entropy` | float ≥ 0 | Entropy of the AI model’s softmax output, representing uncertainty. |
| `user_votes` | list[dict] | A list of user feedback, where each dictionary has: <br> `vote_value` (1 for real, 0 for fake) and `credibility` (float ≥ 0). |

### Outputs
| Output | Type | Description |
|---------|------|-------------|
| `final_score` | float ∈ [−1, +1] | Weighted authenticity score after combining AI and user inputs. |
| `classification` | string | Final decision: `"Real"`, `"Fake"`, or `"Not Sure"`. |

---

## 4. Core Logic and Workflow

### Step 1: AI Component
The AI model outputs `probability_real` from its softmax layer.  
This is mapped into a symmetric scale:
model_score = 2 * probability_real - 1 # range: [-1, +1]

### Step 2: AI Confidence via Entropy
Entropy quantifies uncertainty in the AI output:
H_model = model_entropy # precomputed from AI softmax
ai_confidence = 1 - (H_model / ln(2)) # normalized confidence ∈ [0,1]
w_ai = w_ai_base * ai_confidence # effective AI weight

When the AI output is highly uncertain (entropy near ln(2)), confidence approaches 0; when confident, confidence approaches 1.

---

### Step 3: User Evaluation Component

If fewer than `min_votes` user inputs exist:
final_score = model_score # fallback to AI-only output

Otherwise:
1. Convert user votes into signed values:  
   - Real → +1  
   - Fake → −1  
2. Compute credibility-weighted average:
raw_user_score = Σ(cred_i * vote_i) / Σ(cred_i)

This value lies within [−1, +1].

3. Compute credibility-weighted proportions for “real” and “fake”:
p_real = Σ(cred_i for real votes) / Σ(cred_i)
p_fake = Σ(cred_i for fake votes) / Σ(cred_i)

4. Calculate **user entropy** (natural logarithm base):
user_entropy = - (p_real * ln(p_real) + p_fake * ln(p_fake))

(Omit any term with p = 0.)

5. Normalize entropy and compute confidence:
normalized_entropy = user_entropy / ln(2)
vote_confidence = 1 - normalized_entropy
w_user = w_user_base * vote_confidence

A divided or inconsistent crowd yields high entropy → low weight.

---

### Step 4: Weighted Fusion
Both AI and user inputs are fused proportionally to their **effective weights**:
total_w = w_ai + w_user
final_score = (w_ai / total_w) * model_score + (w_user / total_w) * raw_user_score

This ensures that the most confident component (AI or users) has greater influence.

---

### Step 5: Classification
Thresholding translates the final score into a label:
| Range | Label | Meaning |
|--------|--------|----------|
| ≥ +0.34 | **Real** | Highly likely authentic |
| ≤ −0.34 | **Fake** | Highly likely fake |
| Between −0.34 and +0.34 | **Not Sure** | Uncertain or conflicting evidence |

This middle range introduces a *neutral zone*, improving fairness by preventing overconfident misclassifications.

---

## 5. Entropy and Confidence Interpretation

| Source | Formula | Role |
|---------|----------|------|
| **AI Entropy** | `H_model = −Σ p_i ln(p_i)` | Captures model uncertainty between “real” and “fake.” |
| **AI Confidence** | `1 − (H_model / ln(2))` | Confidence weight applied to AI contribution. |
| **User Entropy** | `H_user = −(p_real ln(p_real) + p_fake ln(p_fake))` | Measures disagreement across credible users. |
| **User Confidence** | `1 − (H_user / ln(2))` | Reduces weight of divided crowds. |

Entropy ensures both model and human uncertainty directly impact decision reliability.

---

## 6. Key Design Principles
Entropy-weighted fairness: The more uncertain component contributes less to the final outcome.

Dynamic adaptability: Works under varying crowd sizes and AI confidence levels.

Transparency: Every decision is explainable through numeric weights and entropy values.

Robustness: Safeguards against unreliable or minimal input conditions.

---

Author: Han Wei Chang
Institution: Taylor’s University – School of Computer Science
