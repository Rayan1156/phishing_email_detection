# Tonight's Build — Complete Step-by-Step Guide
### Phishing Email Detection Project · Everything we did on Mon 2026-11-23, in easy language

**What this project is (in one line):** we built a system that reads an email and decides: **Safe** or **Phishing** (scam). Tonight we finished the entire "brain" of the system — the model. The app comes later.

**The journey in one view:**
Get data → clean it → split it → teach simple models → race them → add letter-features → test hand-made features → combine models into a team → tune how strict the team is → train one big smart model (DistilBERT) → hold a final championship → test the winner once on never-seen emails → save everything.

---

## STEP 0 — Setting up the workshop (Google Colab)

- We work in **Google Colab** (free notebooks in the browser) so we never hit our own laptop's limits — and Colab gives a free GPU later for the big model.
- First check: which tool versions are installed. Result: **pandas 2.2.3** (data tables), **scikit-learn 1.6.1** (the classic ML toolbox). Remember these numbers — when we run the project on a local PC, we install the *same* versions so results match exactly.

## STEP 1 — Getting the dataset

- Source: the Kaggle "Phishing Email Detection" dataset, loaded through its Hugging Face copy (`mriadur/phishing-email-dataset`) because that loads in one line of code.
- Size at arrival: **18,650 emails**.
- **Surprise we caught by actually looking:** the file has only two real columns — `Email Text` and `Email Type` — plus a junk index column. There are **no** separate sender/subject/link columns (our proposal had assumed there were). Every signal lives inside the email text itself. The text also arrives **already lowercased and spaced out** (like `http : / / www . site . com`) — this matters later (Step 9).
- Labels: `Safe Email` and `Phishing Email`. Count: **Safe 11,322 (61%) · Phishing 7,328 (39%)**.

## STEP 2 — Cleaning the data (garbage in, garbage out)

We removed, in order:
1. The junk index column (`Unnamed: 0`) — just row numbers, useless.
2. **16** emails with missing text.
3. **536** junk rows whose entire "email" is literally the word `null`, `empty`, or blank.
4. **~563** exact duplicate emails. *Why this matters:* if the same email sits in both the study material and the exam, the model "cheats" by memory. Duplicates out = honest scores.
5. Converted labels to numbers: **Safe → 0, Phishing → 1**. (Phishing is the "positive" class — the thing we are hunting.)

**Final clean dataset: 17,535 emails** (Safe 10,979 · Phishing 6,556). Saved as `phishing_clean.csv` — the group's one official data file.

## STEP 3 — Splitting: study set, practice set, final exam

We split the 17,535 emails into three piles:

| Pile | Size | Job |
|---|---|---|
| **Train** | 14,027 (80%) | The study material. Models learn only from this. |
| **Validation** | 1,754 (10%) | The practice exam. We compare models and make all decisions here. |
| **Test** | 1,754 (10%) | The final exam. **Locked away, untouched** until the very end. |

- **Stratified** split = every pile keeps the same 37.4% phishing share, so no pile is accidentally easier or harder.
- **`random_state = 42`** = a fixed shuffle seed, so anyone re-running our code gets the *exact same piles*. Reproducibility.
- *Why not just train/test?* Because we planned to make many decisions (which model? which features? which threshold?). If we made those decisions on the test pile, we'd slowly overfit to it. Validation absorbs all the decision-making; the test pile stays pure for one honest final score.

## STEP 4 — How we judge a model (read this part twice)

Four scores, all computed the same way every time:

- **Accuracy** — out of all emails, how many did it get right (both kinds)?
- **Precision** — when it shouts "PHISHING!", how often is it right? Low precision = crying wolf.
- **Recall** — of all the real scams, how many did it catch? **This is our headline score**, because a missed scam can empty someone's bank account, while a false alarm just costs a second look.
- **F1** — one combined number balancing precision and recall.

And the **confusion matrix** — the 2×2 truth table (rows = reality, columns = model's call):

| | Model says Safe | Model says Phishing |
|---|---|---|
| **Really Safe** | TN (correct) | FP (false alarm) |
| **Really Phishing** | **FN (scam slipped through — the dangerous cell)** | TP (caught) |

Every result below can be re-checked from its matrix. We verified every single one by hand tonight.

## STEP 5 — Turning words into numbers (TF-IDF), and the baseline

Machines can't read — they count. **TF-IDF** scores every word by "how common is this word *in this email* vs *across all emails*?" Rare-but-present words (`verify`, `suspended`, `prize`) score high; everywhere-words (`the`, `you`) score near zero. We kept the top **5,000** word features, including 2-word pairs (1–2 grams), so `verify your` counts as its own clue.

**Baseline model — Logistic Regression** (the simplest honest classifier; learns one weight per word):
Accuracy 0.9778 · Precision 0.9753 · **Recall 0.9649** · F1 0.9701 · **23 scams missed** of 656.
A strong start — and our reference point. Every later step must beat this or justify itself.

*(Side lesson from tonight: the model's weights are learned from scratch in `.fit()` — one weight per word-feature, stored in `clf.coef_`. Words like "prize" earn big positive (phishing-side) weights.)*

## STEP 6 — Round 1: four models race on word features

Same features, same data, four different brains (validation results):

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|
| **Linear SVM** | **0.9800** | 0.9755 | **0.9710** | **0.9733** |
| Logistic Regression | 0.9778 | 0.9753 | 0.9649 | 0.9701 |
| Random Forest | 0.9743 | **0.9796** | 0.9512 | 0.9652 |
| Naive Bayes | 0.9658 | 0.9715 | 0.9360 | 0.9534 |

**Findings:** SVM wins Round 1 (~19 scams missed). Random Forest is the most careful shouter (best precision) but misses more scams. Naive Bayes — the simple word-counter — is weakest (recall 0.936, ~42 missed): pure word counting has a ceiling.

## STEP 7 — Round 2: add letter-chunks (character n-grams)

**Idea:** scammers disguise words — `fr33`, `ver1fy`, `acc0unt`. A word-counter sees an unknown word and shrugs. **Character TF-IDF** also reads *chunks of letters* (3–5 characters, `char_wb`), so `ver1fy` still contains chunks like `ver`, `er1`, `1fy` that resemble `verify`'s chunks. We ran the same four models on **words + chunks**:

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| **Linear SVM (word+char)** | **0.9852** | 0.9802 | **0.9802** | **0.9802** |
| Logistic Regression | 0.9806 | 0.9814 | 0.9665 | 0.9739 |
| Random Forest | 0.9766 | 0.9695 | 0.9680 | 0.9687 |
| Naive Bayes | 0.9669 | 0.9476 | 0.9649 | 0.9562 |

**Findings:** chunks helped **every** model — this comparison (Round 1 vs Round 2, changing only the features) is called an **ablation**: change one thing, measure the effect. SVM now misses only **~13 scams** with ~13 false alarms. Naive Bayes gained the most recall (+0.029) — the weakest reader benefits most from better glasses. **SVM (word+char) = champion of the classic models.**

## STEP 8 — Round 3: hand-made checklist features (indicators) — the honest failure

**Idea:** forget words; measure 7 suspicious *behaviours* directly:
1. Link count (`http` + `www` mentions) 2. IP-address pattern (numbers like `192 . 168 . 1 . 1`) 3. Urgency-word hits (`urgent`, `verify`, `suspended`, `prize`, `claim`…) 4. Exclamation count 5. Money symbols (`$`, `£`) 6. Digit ratio 7. Word count.
*(Note: no ALL-CAPS feature — this dataset is pre-lowercased, so that signal doesn't exist here. We said so in the report instead of pretending.)*

Three-way test — indicators **alone**, and **hybrid** (words + chunks + checklist):

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| SVM (hybrid) | 0.9846 | 0.9787 | 0.9802 | 0.9794 |
| LR (hybrid) | 0.9800 | 0.9770 | 0.9695 | 0.9732 |
| LR (indicators only) | 0.7457 | 0.8144 | 0.4146 | 0.5495 |
| SVM (indicators only) | 0.7195 | 0.8417 | 0.3079 | 0.4509 |

**Findings (a real negative result — we report it, we don't hide it):** indicators alone are weak (they'd let 60–70% of scams through), and the hybrid **ties** with words+chunks alone (0.9794 vs 0.9802). Two reasons: (1) the dataset's pre-chewed text destroyed several classic signals (links shattered into spaced tokens, no capital letters); (2) urgency words are *words* — TF-IDF already counts them, so the checklist repeated old news. **Lesson: a feature only helps if it tells the model something new.** The indicators still ship in the app — as the *explanation* panel ("why was this flagged"), where they help humans, and on real Gmail emails (intact links/headers) they carry more meaning.

## STEP 9 — The team photo: soft-voting ensemble

**Idea:** four different readers vote on each email; average their *probabilities* (soft voting) and go with the average. Different models have different blind spots — the vote cancels some out.
**One fix first:** SVM doesn't naturally output probabilities, so we wrapped it in **calibration** (`CalibratedClassifierCV`) — it learns to translate SVM's raw scores into honest percentages. (This also makes the app's "87% risk" mean something real.)

**Result — Accuracy 0.9869 · Precision 0.9803 · Recall 0.9848 · F1 0.9825 · only 10 scams missed, 13 false alarms.**
The ensemble **beat every single model**. New champion.

## STEP 10 — Threshold tuning: choosing how strict to be

A model outputs a *risk percentage*. Someone must choose the line: "flag it if risk ≥ X%." The default X=50 is an accident, not a decision. We swept X on validation:

| Threshold | Precision | Recall | F1 | Missed scams |
|---|---|---|---|---|
| 0.30 | 0.9461 | 0.9909 | 0.9680 | 6 |
| **0.35** | **0.9545** | **0.9909** | 0.9723 | **6** |
| 0.40 | 0.9629 | 0.9893 | 0.9759 | 7 |
| 0.45 | 0.9774 | 0.9878 | 0.9826 | 8 |
| 0.50 (default) | 0.9803 | 0.9848 | 0.9825 | 10 |
| 0.55 | 0.9861 | 0.9741 | 0.9801 | 17 |
| 0.60 | 0.9890 | 0.9634 | 0.9761 | 24 |

Lower line → catch more scams, but more false alarms. **Our decision rule, written down *before* judging:** *the lowest threshold whose precision stays ≥ 0.95* (we refuse to cry wolf more than 1 time in 20; subject to that, catch everything catchable). Rule's answer: **0.35** — 6 missed instead of 10, at ~31 false alarms. (F1 alone peaks at 0.45, but our declared priority is recall.) This 0.35 also becomes the app's default sensitivity.

## STEP 11 — DistilBERT: the heavyweight enters

**What it is, in easy words:** the classic models *count words*. DistilBERT actually *reads* — it was pre-trained on billions of words of English, so it arrives already understanding language, and we **fine-tune** it (a short extra training) to learn one new distinction: scam vs not-scam. It's a smaller, faster version of BERT.

**Setup (with one trap avoided):** DistilBERT needs a **GPU**. Switching the main notebook to GPU would restart it and wipe everything in memory — so DistilBERT got its **own notebook** (`DistilBERT_Experiment`) on a **Tesla T4 GPU**, with the data pipeline re-run identically (same cleaning, same seed 42).

**Training:** emails are chopped into word-pieces by its **tokenizer** (long emails cut at 256 pieces — scam signals live early, so that's fine), then **3 epochs** (3 full passes over the 14,028 training emails), 2,631 steps, **15 minutes 25 seconds**.
- The scary-looking "UNEXPECTED / MISSING" table during loading is **normal**: the pretrained model arrived with a *language* head (discarded) and no *classification* head (created fresh, then trained by us).
- **Loss story:** training loss fell 0.063 → 0.011, but validation loss was *best at epoch 1* (0.046) and crept up (0.060, 0.058) — the first sign of mild **overfitting** (memorizing instead of generalizing). We noted it honestly; the scores below show it didn't cost much.

**Validation result — Accuracy 0.9892 · Precision 0.9907 · Recall 0.9802 · F1 0.9854 · 13 missed, only 6 false alarms.**
Best accuracy, best precision, best F1 of anything so far. But the ensemble still caught more scams (10 missed vs 13). **Split decision** → settled properly in Step 13.

## STEP 12 — The one-email mystery (integrity check)

DistilBERT's confusion matrix totalled **1,753**, not 1,754. We stopped and investigated instead of shrugging: the clean dataset is **identical (17,535)** in both notebooks; the difference is the first split's per-class rounding — the BERT notebook dealt **14,028 train / 1,753 validation**, the main notebook **14,027 / 1,754**. One borderline email sits in a different pile; the validation sets overlap ~99.9%.
**Why this matters:** it means the recall photo-finish (a 2-email gap) is *inside the noise* of that one-row difference — so the championship had to be decided on the wide gaps (F1, precision, false alarms), not the photo-finish. This footnote goes in the report exactly like this.

## STEP 13 — The championship: both finalists at their best settings

Fair rules: each finalist fights at *its own* rule-chosen threshold (the Step 10 rule). DistilBERT's extended sweep (0.05–0.30) revealed its personality: **precision is 0.9878 at every threshold** — it almost never sits on the fence; nearly every email gets a verdict of >95% sure or <5% sure. Its rule-point is **0.05**.

| Finalist @ its tuned threshold | Precision | Recall | F1 | Missed | False alarms |
|---|---|---|---|---|---|
| Ensemble @0.35 | 0.9545 | **0.9909** | 0.9723 | **6** | ~31 |
| **DistilBERT @0.05** | **0.9878** | 0.9878 | **0.9878** | 8 | **~8** |

**Decision (recommended by TARS, confirmed by Rayan): DistilBERT is the champion.** Reasoning: the ensemble catches 2 more scams of 656 (within split noise), but DistilBERT wins F1 by 1.5 points with **a quarter of the false alarms** — and for a tool a human reviews, crying wolf 31 times vs 8 is the difference between a tool that's trusted and one that's ignored. Its decisive scores also make the app's risk meter honest.
**Nothing is wasted:** the ensemble becomes the app's **first-stage screener** in the two-stage **cascade** — the light, instant 25 MB model screens everything; DistilBERT deep-checks. Both finalists ship.

## STEP 14 — The final exam: locked test set, opened exactly once

The champion, at its chosen threshold (0.05), on the 1,754 emails no model had ever seen:

**Accuracy 0.9903 · Precision 0.9878 · Recall 0.9863 · F1 0.9870**
Confusion matrix: TN 1090 · FP 8 · **FN 9** · TP 647.

**What this proves:** test ≈ validation (F1 0.9870 vs 0.9878) — no generalization gap. All our tuning was honest; the headline numbers are real: **99.0% accuracy, 9 scams missed and 8 false alarms in 1,754 unseen emails.** These are the numbers for the report's abstract.
**🏁 MODEL PHASE CLOSED.**

## STEP 15 — Saving everything (so tonight can never be lost)

Colab sessions evaporate — only Drive, downloads, and GitHub are permanent. (One lesson learned the hard way tonight: my first documentation upload to GitHub was double-encoded gibberish — caught by Rayan, fixed, and verified readable.)

| Artifact | What it is | Where it lives |
|---|---|---|
| `ensemble.pkl` (25 MB) | The runner-up team model, trained | PC + Drive `phishing-project/` |
| `distilbert.zip` (~260 MB) | The champion model + tokenizer | PC + Drive `phishing-project/` |
| `phishing_clean.csv` | The clean 17,535-email dataset | PC + Drive |
| `Phishing_Email_Detection.ipynb` | Classic-ML notebook (all rounds) | GitHub repo |
| `DistilBERT_Experiment.ipynb` | BERT training + final test | GitHub repo |
| `Project-Documentation.md` | The formal living documentation | GitHub repo |
| **Repo** | **github.com/Rayan1156/phishing_email_detection** (public) | — |

*(Manual save recipe, for any Colab file: run the save/dump cell → Files panel (folder icon, left edge) → right-click the file → Download. Folders: zip first with `!zip -r name.zip folder`, then download the zip. Notebooks: File → Save a copy in GitHub.)*

---

## All results on one page (validation, unless marked)

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|
| Naive Bayes (words) | 0.9658 | 0.9715 | 0.9360 | 0.9534 |
| Random Forest (words) | 0.9743 | 0.9796 | 0.9512 | 0.9652 |
| Logistic Regression (words) — baseline | 0.9778 | 0.9753 | 0.9649 | 0.9701 |
| Linear SVM (words) | 0.9800 | 0.9755 | 0.9710 | 0.9733 |
| SVM (indicators only) | 0.7195 | 0.8417 | 0.3079 | 0.4509 |
| LR (indicators only) | 0.7457 | 0.8144 | 0.4146 | 0.5495 |
| SVM (hybrid) | 0.9846 | 0.9787 | 0.9802 | 0.9794 |
| Linear SVM (word+char) | 0.9852 | 0.9802 | 0.9802 | 0.9802 |
| Ensemble @0.50 | 0.9869 | 0.9803 | 0.9848 | 0.9825 |
| Ensemble @0.35 (tuned) | — | 0.9545 | 0.9909 | 0.9723 |
| DistilBERT @0.50 | 0.9892 | 0.9907 | 0.9802 | 0.9854 |
| **DistilBERT @0.05 (champion)** | — | **0.9878** | **0.9878** | **0.9878** |
| **DistilBERT @0.05 — FINAL TEST** | **0.9903** | **0.9878** | **0.9863** | **0.9870** |

## Mini-glossary (tonight's words, plain meanings)

- **TF-IDF** — scoring words by how *distinctive* they are, not just how often they appear.
- **N-gram** — a chunk: 1–2 *words* (word n-grams) or 3–5 *letters* (character n-grams).
- **Ablation** — changing exactly one thing between runs to learn what that thing does.
- **Calibration** — teaching a model's raw scores to behave like honest percentages.
- **Threshold** — the risk line: flag everything at or above it.
- **Epoch** — one full pass over all training data.
- **Fine-tuning** — taking a model that already knows language and teaching it one new job.
- **Stratified** — splitting while keeping the class mix identical in every pile.
- **Overfitting** — memorizing the study material instead of learning the subject; great on train, weaker on new data.
- **Cascade** — a fast model screens everything; a heavy model re-checks — speed *and* accuracy.

## What remains (from tomorrow morning)

1. **Adversarial test** — take scams the champion catches, disguise them (leetspeak `fr33`, spaced letters `v e r i f y`, look-alike domains), and re-measure: does a cheap disguise defeat it?
2. **Local test set** — ~30 phishing emails written by our own group (JazzCash/Easypaisa lures, bank alerts, prize scams — English + Roman Urdu) + ~15 safe ones, also planted in a demo Gmail inbox. *Test fixtures only — never sent to real people.*
3. **Error analysis** — read the champion's few misses and false alarms; document the patterns.
4. **The app (Streamlit)** — Stage 1: paste-an-email mode. Stage 2: explanation panels (why-panel, link inspector, sender check, urgency detector with the Pakistani-scam pack). Stage 3: Gmail connect (read-only) so the teacher can scan a real inbox. Plus the README polish in the repo.
