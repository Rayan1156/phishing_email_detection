# AI Semester Project — Phishing Email Detection
## Project Documentation (living document — updated after every build step)

**Team:** Muhammad Rayan Bin Waqar, Abdullah Sajid, Muhammad Mujtaba, Muzamil Shafique, Husnain, Muhammad Hassaan
**Submitted to:** Noor Fatima
**Platform:** Google Colab (training) · Streamlit (app, later stage)
**Repository:** https://github.com/Rayan1156/phishing_email_detection (public; notebook saved from Colab; this documentation lives in the repo)
**Status:** Model auditions in progress (Step 8 of the build)

---

## 1. Project Overview

A machine-learning system that classifies emails as **Safe** or **Phishing**, built in three layers beyond a standard dataset project:

1. **Model layer** — a fair, documented comparison of classic ML models, an ensemble, and a fine-tuned transformer (DistilBERT) on identical data splits.
2. **Robustness layer** — an adversarial obfuscation test (disguised scam emails) and a self-made local test set of Pakistani scam emails (JazzCash/Easypaisa lures, Roman Urdu), written by the team as test fixtures only.
3. **Product layer** — a working app with three input modes (paste an email, upload an .eml file, connect a Gmail account read-only) so the teacher can test the system on a real inbox, with risk scores, explanation panels, link inspection, and sender checks.

## 2. Dataset

- **Source:** "Phishing Email Detection" dataset (Kaggle), loaded via its Hugging Face mirror `mriadur/phishing-email-dataset` for reproducibility in Colab.
- **Raw size:** 18,650 rows.
- **Columns (as actually found):** `Email Text`, `Email Type` — plus a junk index column (`Unnamed: 0`). There are **no** separate sender/subject/URL columns; all signals live inside the email text. Text arrives already lowercased and tokenized.
- **Raw class balance:** Safe Email 11,322 (60.7%) · Phishing Email 7,328 (39.3%).

## 3. Data Cleaning (Step 4 — done, verified)

| Action | Rows affected |
|---|---|
| Dropped junk index column (`Unnamed: 0`) | — |
| Dropped missing email texts | 16 |
| Dropped junk rows whose text is literally `null` / `empty` / blank | 536 |
| Dropped exact duplicate emails (prevents train/test leakage) | ~563 |
| **Final clean dataset** | **17,535 rows** |

- Final balance: Safe 10,979 (62.6%) · Phishing 6,556 (37.4%).
- Labels mapped: Safe Email → 0, Phishing Email → 1 (phishing is the positive class).
- Clean dataset exported as `phishing_clean.csv` (group's shared file of record).

## 4. Data Split (Step 5 — done, verified)

**80 / 10 / 10 — Train / Validation / Test**, stratified (identical 37.4% phishing share in all three), `random_state = 42` for reproducibility.

| Set | Rows | Role |
|---|---|---|
| Train | 14,027 | Models learn from this |
| Validation | 1,754 | Model selection, threshold tuning, all comparisons |
| Test | 1,754 | **Locked** — touched once, for the final reported scores |

*Methodology note:* model selection is done on validation only, so the test set stays an unbiased final exam. DistilBERT fine-tuning also requires a validation set (checkpoint selection / early stopping).

## 5. Evaluation Metrics

Accuracy, Precision, Recall, F1, and the confusion matrix — all computed on validation during development.
**Headline metric: Recall** (fraction of phishing emails caught), because a missed scam (false negative) costs more than a false alarm. The dangerous confusion-matrix cell is FN (bottom-left).

## 6. Feature Engineering Plan

- **Round 1:** word TF-IDF (1–2 grams, 5,000 features).
- **Round 2:** word TF-IDF + character TF-IDF (`char_wb`, 3–5 grams, 5,000 features) — character chunks catch obfuscated words (`fr33`, `ver1fy`).
- **Round 3 (planned):** handcrafted indicator features fused with TF-IDF — URL count and URL lexical features (length, digit ratio, subdomain depth, entropy, `@` tricks), urgency-keyword count, ALL-CAPS ratio, exclamation density, generic-greeting flag, and a **brand-impersonation feature** (fuzzy distance between link domains and known brands, incl. Pakistani banks/wallets).
- **Ablation:** text-only vs indicators-only vs hybrid, to show which signal carries the detection.

## 7. Results So Far (Validation Set)

### Baseline — Logistic Regression, word TF-IDF (Step 6)
Accuracy 0.9778 · Precision 0.9753 · Recall 0.9649 · F1 0.9701
Confusion matrix: TN 1082 · FP 16 · FN 23 · TP 633 (23 scams missed of 656)

### Round 1 — word TF-IDF, four models (Step 7)
| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| **Linear SVM** | **0.9800** | 0.9755 | **0.9710** | **0.9733** |
| Logistic Regression | 0.9778 | 0.9753 | 0.9649 | 0.9701 |
| Random Forest | 0.9743 | **0.9796** | 0.9512 | 0.9652 |
| Naive Bayes | 0.9658 | 0.9715 | 0.9360 | 0.9534 |

*Findings:* Linear SVM leads round 1. Random Forest has the best precision (fewest false alarms) but weaker recall. Naive Bayes is the weakest detector (recall 0.936 — ~42 scams missed): simple word counting has a ceiling.

### Round 2 — word + char TF-IDF (Step 8)
| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| **Linear SVM (word+char)** | **0.9852** | 0.9802 | **0.9802** | **0.9802** |
| Logistic Regression (word+char) | 0.9806 | 0.9814 | 0.9665 | 0.9739 |
| Random Forest (word+char) | 0.9766 | 0.9695 | 0.9680 | 0.9687 |
| Naive Bayes (word+char) | 0.9669 | 0.9476 | 0.9649 | 0.9562 |

*Ablation finding (Round 1 → Round 2):* adding character n-grams improved **every** model's recall. SVM: F1 0.9733 → 0.9802, missed scams 19 → ~13 (FP also ~13). Naive Bayes recall jumped 0.9360 → 0.9649 — the letter-chunk features help the weakest reader most. Conclusion so far: character features earn their place; Linear SVM (word+char) is the current champion.

### Round 3 — handcrafted indicators + hybrid (Step 9)
| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| SVM (hybrid) | 0.9846 | 0.9787 | 0.9802 | 0.9794 |
| LR (hybrid) | 0.9800 | 0.9770 | 0.9695 | 0.9732 |
| LR (indicators only) | 0.7457 | 0.8144 | 0.4146 | 0.5495 |
| SVM (indicators only) | 0.7195 | 0.8417 | 0.3079 | 0.4509 |

*Ablation conclusion (complete three-way: text-only / indicators-only / hybrid):* on **this** dataset, the seven handcrafted indicators add **nothing** on top of word+char TF-IDF (hybrid F1 0.9794 ≈ text-only 0.9802 — a statistical tie), and alone they are weak detectors (recall 0.31–0.41: they miss most scams). Two reasons: (1) the dataset's text is pre-tokenized and lowercased, destroying several classic indicator signals (intact URLs, casing); (2) signals like urgency words are already captured by the 10,000 TF-IDF features, so the checklist adds no *new* information. **This is a genuine negative result and is reported as such.** The indicators still ship in the app — as the *explanation* layer ("why" panel) and for real emails, whose intact URLs/headers carry signals this dataset has already destroyed. **Champion remains Linear SVM (word+char), F1 0.9802.**

### Ensemble — soft voting, word+char (Step 10)
Members: calibrated Linear SVM + Logistic Regression + Random Forest + Naive Bayes, soft (probability) voting.
**Accuracy 0.9869 · Precision 0.9803 · Recall 0.9848 · F1 0.9825** — Confusion matrix: TN 1085 · FP 13 · FN 10 · TP 646.

*Finding:* the ensemble **beats every single model** — only **10 of 656 scams missed** (champion SVM missed 13) at the same 13 false alarms. Averaging four different readers cancels individual blind spots. **New champion: the ensemble.** Its calibrated probabilities also power the app's risk percentage and the threshold tuning that follows.

### Threshold tuning (Step 11) — operating point LOCKED at 0.35
Sweep on validation probabilities (ensemble):

| Threshold | Precision | Recall | F1 | Missed scams |
|---|---|---|---|---|
| 0.30 | 0.9461 | 0.9909 | 0.9680 | 6 |
| **0.35** | **0.9545** | **0.9909** | 0.9723 | **6** |
| 0.40 | 0.9629 | 0.9893 | 0.9759 | 7 |
| 0.45 | 0.9774 | 0.9878 | 0.9826 | 8 |
| 0.50 (default) | 0.9803 | 0.9848 | 0.9825 | 10 |
| 0.55 | 0.9861 | 0.9741 | 0.9801 | 17 |
| 0.60 | 0.9890 | 0.9634 | 0.9761 | 24 |

*Decision rule (stated before seeing results): the lowest threshold whose precision stays ≥ 0.95.* → **0.35**: catches 650/656 scams (99.1%) at ~31 false alarms, vs 10 missed at the default 0.50. F1 is marginally higher at 0.45, but the project's stated priority is recall — a missed scam costs more than a false alarm — so 0.35 is the official operating point and the app's default sensitivity.

### DistilBERT fine-tune (Step 12) — validation
3 epochs, T4 GPU, 15 min 25 s, max_length 256, identical splits (seed 42). Training loss 0.063 → 0.011; validation loss best at epoch 1 (0.046), rising slightly after (0.060, 0.058) — mild overfitting after epoch 1.
**Accuracy 0.9892 · Precision 0.9907 · Recall 0.9802 · F1 0.9854** — Confusion matrix: TN 1092 · FP 6 · FN 13 · TP 642.

*Finding:* DistilBERT takes the **F1 and precision crowns** (only 6 false alarms — half the ensemble's 13) and the best accuracy, but the ensemble keeps the **recall crown** (10 missed vs 13). The championship is therefore decided at *tuned* operating points: the ensemble @0.35 gives P 0.9545 / R 0.9909 (6 missed); DistilBERT's threshold sweep is run next under the same rule.
*Data note:* the DistilBERT notebook's validation matrix totals 1,753 vs 1,754 in the main notebook — a one-row difference in the rebuilt pipeline, flagged for a shape check; it does not change any conclusion (all metrics verified internally consistent).

### Championship — tuned operating points (validation)
DistilBERT's extended sweep (0.05–0.30) showed its probabilities are *decisive*: precision is constant at 0.9878 across the whole range — almost no email scores in the uncertain middle. Its rule-chosen point is **0.05: P 0.9878 · R 0.9878 · F1 0.9878 · 8 missed · ~8 false alarms**.

| Finalist @ tuned threshold | Precision | Recall | F1 | Missed | False alarms |
|---|---|---|---|---|---|
| Ensemble @0.35 | 0.9545 | **0.9909** | 0.9723 | **6** | ~31 |
| **DistilBERT @0.05** | **0.9878** | 0.9878 | **0.9878** | 8 | **~8** |

*Decision:* the ensemble catches 2 more scams of 656 (recall +0.3 pts); DistilBERT wins F1 by 1.5 pts with a quarter of the false alarms and far more decisive scores. **Champion: DistilBERT @ threshold 0.05** — CONFIRMED by Rayan 2026-11-23, with the ensemble as runner-up and the app's fast first-stage screener in the cascade design. The locked test set is opened once, for the champion's final confirmation score.

### FINAL TEST RESULTS — locked test set, opened once (Step 13)
**DistilBERT @ threshold 0.05 — Accuracy 0.9903 · Precision 0.9878 · Recall 0.9863 · F1 0.9870**
Confusion matrix: TN 1090 · FP 8 · FN 9 · TP 647 (of 1,754 unseen emails: 9 scams missed, 8 false alarms).

*Finding:* the test scores match validation almost exactly (F1 0.9870 vs 0.9878) — no generalization gap; the champion's validation numbers were honest. **The model phase of the project is closed.** Headline for the report: *99.0% accuracy, F1 0.987 on 1,754 never-seen emails.*

## 8. Planned Enhancements (agreed 2026-11-23)

- **Threshold tuning** on validation (precision–recall curve): deliberately trade a little precision for higher recall; the app exposes this as a sensitivity control.
- **Probability calibration** so the app's risk percentage is meaningful.
- **Ensemble** (soft voting over the top classic models).
- **DistilBERT fine-tune** (3 epochs, Colab GPU) on the same splits; final comparison table classic vs transformer.
- **Two-stage cascade in the app:** fast classic model screens everything; only uncertain cases (mid-range risk) go to DistilBERT.
- **App-only signals:** Gmail authentication results (SPF/DKIM/DMARC) as verdict evidence — unavailable to dataset-only projects.
- **Correction loop:** teacher overrides in the app are logged as future training data.

## 9. Adversarial & Local Testing (planned)

- **Adversarial test:** take phishing emails the champion model catches; obfuscate them (leetspeak, spaced letters, homoglyph/look-alike domains); re-measure recall. Question: does a cheap disguise defeat the detector, and do char features defend it?
- **Local test set:** ~30 team-written phishing emails (JazzCash/Easypaisa, bank lures, prize scams; English + Roman Urdu) + ~15 team-written safe emails. Also planted in a demo Gmail inbox for the live app demonstration. *These are test fixtures, sent only to the team's own test account — never to real people.*
- **Error analysis:** read the champion's false negatives and false positives; document the failure patterns.

## 10. Application (planned — after the model is locked)

Streamlit app, three input modes:
1. **Paste email text** → instant verdict.
2. **Upload .eml** → verdict with full indicator panels.
3. **Connect Gmail** (read-only OAuth; runs locally; nothing stored) → scan recent inbox, sort by risk, CSV export.

Panels: risk meter with bands · "Why" panel (top contributing words highlighted) · link inspector (real destination domains, look-alike flags) · sender check (name/domain mismatch) · urgency/lure detector incl. Pakistani-scam keyword pack · inbox dashboard.

## 11. Build Log

| Date | Step | Outcome |
|---|---|---|
| 2026-11-19 | Proposal delivered | Word proposal to Noor Fatima (dataset correction noted separately) |
| 2026-11-23 | Scope upgraded | Gmail app + upgraded ML + adversarial/local testing agreed |
| 2026-11-23 | Steps 1–3 | Colab set up (pandas 2.2.3, sklearn 1.6.1); dataset loaded and inspected |
| 2026-11-23 | Steps 4–5 | Cleaned to 17,535 rows; 80/10/10 stratified split frozen |
| 2026-11-23 | Steps 6–7 | LR baseline + round-1 race complete (SVM leads) |
| 2026-11-23 | Step 8 | Round 2 (word+char) running |

## 12. Saving the Project & Running It Locally

**Golden rule:** everything inside Colab's `/content` **vanishes when the session ends**. Only three things are permanent: Google Drive, downloaded files, GitHub.

**Save from Colab:**
1. Notebook: `File → Save a copy in Drive` (keeps the .ipynb in Drive). Extra backup: `File → Download → .ipynb`.
2. Project files (dataset, models): mount Drive once per session —
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
   and save into `/content/drive/MyDrive/phishing-project/` (create the folder in Drive first). Copy `phishing_clean.csv` there.
3. Trained models: `joblib.dump(model, "name.pkl")` — because our models are sklearn **Pipelines**, one .pkl holds the TF-IDF vectorizer + classifier together, ready for the app.
4. Group-friendly option: `File → Save a copy in GitHub` into a repo — version history + all members can clone it.

**Run locally (Windows):**
```bat
python -m venv venv
venv\Scripts\activate
pip install pandas==2.2.3 scikit-learn==1.6.1 streamlit joblib
```
- Versions pinned to match the Colab environment used for training (pandas 2.2.3, scikit-learn 1.6.1) so results reproduce exactly.
- Notebook work: open the .ipynb in VS Code / PyCharm / Jupyter with `phishing_clean.csv` in the same folder.
- The final app: `streamlit run app.py` → opens at `localhost:8501` in the browser. The app loads the champion model's .pkl — no retraining needed on the teacher's PC beyond installing the requirements.
