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
