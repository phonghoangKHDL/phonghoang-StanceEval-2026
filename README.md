# phonghoang at StanceEval-2026: A Two-Stage Framework for Arabic Stance Detection

This repository contains the source code for Team phonghoang's submission to Track 1 (Seen Targets) of the StanceEval-2026 Shared Task on Arabic Stance Detection.

Our system addresses the challenges of in-domain optimization and cross-target generalization via a two-stage framework:

1. **Stage 1 (Development Phase):** A Multi-Task MARBERTv2 architecture (stance, sentiment, sarcasm) trained with 10-Fold Stratified Cross-Validation, further regularized with R-Drop and Stochastic Weight Averaging (SWA).
2. **Stage 2 (Testing Phase):** An LLM-guided data augmentation strategy using Gemini 3.6 Flash for contrastive few-shot pseudo-labeling, fused with the external ArabicStanceX dataset and high-confidence source-domain samples, to adapt to the related test target ("Women Driving").

## System Performance

### Table 1: Development Set Ablation

| Model Configuration | Dev F_avg2 |
|---|---|
| MARBERT (Focal Loss + Threshold Tuning) | 0.8302 |
| MARBERT (Multi-Task Learning) | 0.8250 |
| AraBERT (10-Fold CV + Margin) | 0.7923 |
| MARBERT+AraBERT Ensemble | 0.8366 |
| MARBERT (10-Fold + Multi-Task) | 0.8527 |
| MARBERT (10-Fold + R-Drop + SWA) | 0.8501 |
| **Final Dev Ensemble (2:1 Weighted Blend)** | **0.8561** |

### Table 2: Stage 2 Ablation on the Official Test Set (target: Women Driving)

| Stage 2 Configuration | Test F_avg2 |
|---|---|
| Stage 1 ensemble (no adaptation) | 0.6701 |
| + LLM few-shot annotation | 0.8310 |
| + LLM pseudo-labeling (retrain) | 0.8382 |
| + External data fusion (ArabicStanceX) | **0.8431** |

For detailed ablation studies and error analysis, please refer to our system description paper.

## Repository Structure

```text
.
├── data/                                      # Not included, see note below
├── model/                                      # Not included, see note below
├── notebooks/
│   ├── download_model.ipynb                    # Utility: download/convert MARBERTv2 base weights
│   ├── 02_FocalLoss_and_ThresholdTuning.ipynb   # Stage 1: Focal loss baseline + dev-set threshold tuning
│   ├── 03_MultiTask_Learning.ipynb              # Stage 1: single-model Multi-Task MARBERT baseline
│   ├── 03b_MultiTask_PerTarget.ipynb            # Stage 1 (exploratory): per-target expert models
│   ├── 06_Fold_Cross_Validation.ipynb           # Stage 1: 10-Fold CV, Multi-Task MARBERT, OOF probs
│   ├── 08_RDrop_SWA.ipynb                       # Stage 1: 10-Fold CV with R-Drop + SWA, and the
│   │                                             #          final 2:1 weighted blend (Final Dev Ensemble)
│   ├── 09_AraBERT_Large_10Fold.ipynb            # Stage 1 (ablation): AraBERT 10-Fold CV baseline
│   ├── 10_Mega_Ensemble.ipynb                   # Stage 1 (ablation): MARBERT+AraBERT soft-voting ensemble
│   └── PhaseTest.ipynb                          # Stage 2: data augmentation, retraining, test submission
├── evaluate.py                                  # Official evaluation metric script
├── phonghoang_StanceEval_2026.pdf               # System description paper
├── README.md
└── requirements.txt
```

> **Note:** The `data/` and `model/` directories are not included in this repository. Download the official Mawqif-v2 dataset from the StanceEval-2026 organizers, the external ArabicStanceX dataset from Hugging Face, and the MARBERTv2 base weights via `download_model.ipynb`.

## How to Run

### 1. Set Up Environment

```bash
python -m venv .venv
source .venv/bin/activate        # On Windows: .venv\Scripts\activate
```

### 2. Install Dependencies

All required packages, including PyTorch, Transformers, Datasets, scikit-learn, and the Google GenAI client (`google-genai`), are listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 3. Data and Base Model Preparation

Place the official Mawqif-v2 dataset files into the `data/` folder. Run `download_model.ipynb` to download and convert the MARBERTv2 base checkpoint into `model/marbert_base/`.

### 4. Stage 1: In-Domain Optimization

Run the notebooks in `notebooks/` in the following order:

1. `02_FocalLoss_and_ThresholdTuning.ipynb` and `03_MultiTask_Learning.ipynb` for the single-model baselines.
2. `06_Fold_Cross_Validation.ipynb` to train the 10-Fold Multi-Task MARBERT models and export their out-of-fold (OOF) and test probabilities.
3. `08_RDrop_SWA.ipynb` to train the 10-Fold R-Drop + SWA models, export their OOF and test probabilities, and compute the Final Dev Ensemble (2:1 weighted blend).
4. Optionally, `09_AraBERT_Large_10Fold.ipynb` and `10_Mega_Ensemble.ipynb` reproduce the AraBERT baseline and the MARBERT+AraBERT ensemble ablation reported in Table 1.

### 5. Stage 2: Cross-Target Adaptation (Pseudo-Labeling)

Insert your Google Gemini API key into `PhaseTest.ipynb`. Running the notebook will:

- Prompt Gemini 3.6 Flash for pseudo-labels on the test target ("Women Driving") using the contrastive few-shot prompt described in the paper's appendix.
- Fuse these pseudo-labels with the ArabicStanceX external data and the high-confidence source-domain samples into the augmented training set.
- Train the final 5-Fold model ensemble on the augmented data and produce the test-set submission file.

## Citation

If you use this code or methodology, please cite our system description paper:

```bibtex
@inproceedings{phonghoang-stanceeval2026,
  title     = {phonghoang at StanceEval-2026: A Two-Stage Framework for Arabic Stance Detection via Multi-Task Regularization and LLM-Guided Data Augmentation},
  author    = {Dinh, Hoang Phong and Dang, Van Thin},
  booktitle = {Proceedings of the 4th Arabic Natural Language Processing Conference (ArabicNLP 2026)},
  address   = {Budapest, Hungary},
  publisher = {Association for Computational Linguistics},
  year      = {2026},
  note      = {System description paper for the StanceEval-2026 Shared Task Track 1}
}
```