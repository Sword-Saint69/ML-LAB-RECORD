# ML Lab Record — Rules & Constraints

Running rules for the PCCSL 507 Machine Learning lab record (LaTeX). Follow these on every experiment unless the user overrides them.

## 1. Document setup (from the .docx template, keep same structure)
- Paper: US Letter. Margins: top 0.7in, bottom 0.63in, left/right 0.75in.
- Font: Times clone (`mathptmx`), 11pt base. Section headings: bold 14pt, **left-aligned** (template style). No `Title of the experiment:` prefix — bare title text only. Never use short forms SLR/MLR — always Simple/Multiple Linear Regression.
- Colors: primary `#1F3864`, grey text `#555555`, table grid `#999999`.
- Header (all pages): `PCCSL 507 Machine Learning Laboratory Record`, bold primary, primary bottom rule.
- Footer (all pages): `DEPARTMENT OF CSE ... CAPE COLLEGE OF ENGINEERING ALAPPUZHA ... Page No. <n>`. Experiment header shows Date only (no Page No. blank at top).
- Cover institution line: `CAPE COLLEGE OF ENGINEERING ALAPPUZHA`.
- Cover details (Name … Faculty In-Charge, Signature/Date): centered block, fixed-width label column, vertically aligned colons and write-in lines.
- Index table: header row dark-blue fill + white bold text; fill each experiment's title + date + page number. Every date (index + sheet Date lines) uses d-m-yyyy with dashes and a 4-digit year. Own dates: 1A+1B 7-7-2026, 2 15-7-2026, 3A+3B 22-7-2026, 4A+4B+4C 14-8-2026, 5+6 9-9-2026; filled pic dates for 7-11 (see Dates bullet in Files).

## 2. Code rules (strict)
- **No scikit-learn in exps 1A-4B except Experiment 4C** (no `fetch_*`, no `train_test_split`, no metrics elsewhere). NumPy + stdlib only; `matplotlib` allowed for plots. Exps 5-6 use the record's own code (sklearn allowed there); exps 7-11 follow the official procedure sheets (sklearn/keras as they require).
- 4C exception: uses `fetch_california_housing` + `LinearRegression` (X=AveRooms, 80/20 `random_state=42`).
- Data always comes from the **`housing.csv` file** (California Housing), never a library loader.
- Experiment 1 fixed spec: X = TotalRooms, Y = MedHouseVal (in $100k), 80/20 split with seed 42, standardize with train mean/std.
- Least squares via classic formulas; gradient descent with m = c = 0, α = 0.01, 100 epochs, MSE cost `J = 1/n Σ(Y−Ŷ)²`, gradients with −2/n factor.
- Metrics: MSE, R² (and RMSE). Real run output only — never invent numbers.
- No `#` comments in any program or listing. Listings must match their `.py` file exactly.
- Code style: simple, beginner-readable, faculty notation (`m`, `b`), plain statements with no comments.

## 3. Section content rules (Experiment 1 pattern)
- TITLE: short, filled in. AIM first, then OBJECTIVE (2–3 lines each).
- THEORY: equations only (regression equation, LS estimates, MSE cost, gradients, updates, MSE/R²).
- PROCEDURE: faculty style — Part A least squares steps + Part B gradient-descent steps.
- DATASET DESCRIPTION: real counts (usable/missing/duplicates/invalid, train/test sizes, feature, target).
- PROGRAM: full listing, must match its `.py` file exactly. OUTPUT: real console output + plots (sales exp has bar/pie/line/scatter in a 2x2 grid). RESULT: always the exact template sentence with the model-name blank filled in **bold, no underline**.
- Exp 4A/4B/4C use the full faculty procedures (least-squares B–I, gradient-descent 1–14, sklearn 1–6).

## 4. Double-side printing layout
- Document class: `twoside`.
- **LEFT (even) pages: PROGRAM code pages only.**
- **RIGHT (odd) pages: everything else** — cover, index, Title/Theory sheet, outputs, result.
- Split code by method if needed so each code page fits one left page: PROGRAM (A) / OUTPUT (A) / PROGRAM (B) / OUTPUT (B). Single PROGRAM covering all methods is preferred when it fits the parity scheme (code starts on a left page, may span two pages).
- Blank parity pages are allowed; they must use `\thispagestyle{empty}` (no header/footer) but keep page numbering continuous.
- No section may spill onto the wrong side — reflow (wording, figure widths, listing length) until each section starts on its correct side, verified via `pdftotext` page check after every compile.

## 5. Files (in `ML LAB` folder)
- `ML_Lab_Record.tex` / `.pdf` — the joined record (cover, index, all 6 experiments, one PDF).
- `OPWER.tex` / `.pdf` — big-code-font variant of the joined record (PCCSL 508 header; code AND output on left pages).
- Joined order: 1A marks, 1B dataframe, 2 sales (+bar+pie charts), 3A hours-score, 3B MLR, 4A LS, 4B GD, 4C sklearn. Index page numbers must match section starts; verify with `pdftotext` after every compile.
- `ML_Lab_Experiment1.tex` / `.pdf` — full record: cover, index, experiment.
- `ML_Lab_Report_Template.tex` — blank template (program/output pages before experiment sheet).
- `exp1_run.py` — the exact program in the listing. `exp1_ls.png`, `exp1_cost.png` — the figures.
- `ML_Lab_Experiment2.tex` / `.pdf` — Experiment 2 (NumPy marks array), same template. `exp2_run.py` — its program.
- `ML_Lab_Experiment3.tex` / `.pdf` — Experiment 3 (pandas student dataframe + CSV). `exp3_run.py`, `exp3_students.csv`.
- `ML_Lab_Experiment4.tex` / `.pdf` — Experiment 4 (NumPy 2D sales array). `exp4_run.py`.
- `ML_Lab_Experiment5.tex` / `.pdf` — Experiment 5 (SLR hours vs score, predict at 9). `exp5_run.py`, `exp5_fit.png`.
- `ML_Lab_Experiment6.tex` / `.pdf` — Experiment 6 (MLR area+age, 3D plane). `exp6_run.py`, `exp6_plane.png`.
- `ML_Lab_Experiment7.tex` / `.pdf` — Experiment 7 (KNN Fashion MNIST, from-scratch NumPy). `exp7_run.py`, `exp7_acc.png`.
- `OPWER.tex` / `.pdf` — joined record, big code font, PCCSL 508 header, code+output LEFT (15 exps: 1A,1B,2,3A,3B,4A,4B,4C,5,6,7,8,9,10,11). Index pages verified with pdftotext.
- Cover: emblem `cape_logo.jpg`, "Department of Computer Science and Engineering", filled (Name Goutham Sankar JL, Reg PRP24CS068, B.Tech, S5, 2026-2027, Faculty Ms. Sneha S, Ms. Deepa A B). Cover title says PCCSL 508.
- Dates: exps 1A-6 keep the record's own dates (d-m-yyyy dashes, 4-digit year: 7-7-2026, 15-7-2026, 22-7-2026, 14-8-2026, 9-9-2026). Only where dates were missing, pic dates were filled into index AND sheet Date lines: 7/8 07-10-2026, 9/10 14-10-2026, 11 30-10-2026.
- Exps 10 & 11 (user's pic titles): 10 "K-Means Clustering on Digits Dataset" (best K=15 by silhouette; `exp10_run.py`, `exp10_elbow.png`, `exp10_sil.png`), 11 "Multilayer Feed-Forward Neural Network (MLP) on Wine Quality Dataset using Keras" (`exp11_run.py`, `exp11_acc.png`). Program pages split A/B after `model.compile(...)` line.
- `winequality-red.csv` — UCI red wine (1599 rows, sep=";"), exp11 dataset; `cape_logo.jpg` — CAPE emblem (copied from Projects/MOCHI/docs/emblem.jpg).
- Keras runs on the torch backend (`os.environ["KERAS_BACKEND"]="torch"` before importing keras; system has torch 2.14, no TF). Installed into `.venv`: keras 3.15.1.
- `mastercopy-1.pdf` (Downloads) — faculty master copy for exps 5-9. OPWER exps 7-9 follow it (titles/theory/procedures). Exps 5-6 stay the record's own versions (user choice on 24/10/26); exp 6 = tercile segmentation per customer RFM features.
- Official procedure sheets: user's pic `Downloads/EXPERIMENT 7_5_14_11.pdf` (pdftotext before use). Content mapping to OPWER: PDF exp7 -> 7, exp5 (Ridge/Lasso) -> 8, exp6 (MLE/MAP) -> 9, exp14 (K-Means Digits) -> 10, exp11 (MLP Wine) -> 11. Procedures applied VERBATIM (7: numbered 7 steps; 8: numbered with Train sub-list; 9: prose lines + bullet sub-lines; 10: unnumbered 9 lines; 11: numbered 8 steps with sub-descriptions). Extraction artifacts cleaned: "XX and yy" -> "X and y", "K=2K=2 to K=15K=15" -> "K=2 to K=15". Official exp11 procedure says "Dense(1, softmax)" (their typo); real run + listing use Dense(6) (6 quality classes) — user approved keeping the sheet text verbatim.
- `exp7_run.py` — Pima Indian Diabetes (diabetes.csv) Logistic Regression ± StandardScaler. `diabetes.csv` — Pima dataset (plotly mirror, 768x9).
- `exp8_run.py` — OLS/Ridge/Lasso on sklearn diabetes; `exp8_pred.png`, `exp8_coef.png`.
- `exp9_run.py` — Breast Cancer LR via MLE (C=inf), MAP L2, MAP L1 (saga, l1_ratio=1); no figures.
- `exp5_knn_run.py`, `exp5_knn_acc.png`, `exp6_tree_run.py`, `exp6_tree.png`, `exp6_imp.png`.
- `online_retail.zip` — UCI Online Retail, git-ignored, exp6 auto-downloads it. `.venv` — system-site venv with pandas/openpyxl (system python lacks them).
- Pandas is allowed where the experiment needs DataFrames (Exp 3); ML model code stays NumPy-only.
- exp11_run.py (official-procedure build): Dense(32, relu) / Dense(16, relu) / Dense(6, softmax), epochs=150, batch_size=100, stratified 80/20 (random_state=42), labels 3-8 -> 0-5. Real rerun output (verbatim in doc): Dataset (1599, 12), Missing values 0, Duplicates 240, Train 1279 Test 320, train acc 0.6935, test loss 0.9831, test acc 0.6281. `exp11_acc.png` regenerated from that run.
- Sheet-fit (exps 9/10/11 each on one page): the sheet's `{...}` group holds `\makeatletter\let\vspace\@gobble\makeatother` + parskip/display-skip 0pt + merged short desc lines (9: Name+Source and Features+Target; 10: Name+Source; 11: Name+Source, plus the Description tail shortened — the label-shift info already lives in its Classes line). Exp 11 additionally gets `\fontsize{10}{11.4}\selectfont` in its group. Result: 103 pages, all 15 index rows == section starts (verified), parity audit clean on pages 5-103.
- Listing audit (OPWER): all 20 lstlisting blocks byte-match their `.py` (15 exp slots; exps 2/6/8/11 are split into 2 blocks each). Mapping: 1A<-exp2_run.py, 1B<-exp3_run.py, 2<-exp4_run.py, 3A<-exp5_run.py, 3B<-exp6_run.py, 4A<-exp4a_run.py, 4B<-exp4b_run.py, 4C<-exp4c_run.py, 5<-exp5_knn_run.py, 6<-exp6_tree_run.py, 7-11<-expN_run.py; exp1_run.py belongs to ML_Lab_Experiment1 (not OPWER). exps 2/3A/3B code files (exp4/5/6_run.py) were trimmed to the doc listings (X-/Y-echo prints, "saved ..." prints, 1 blank line removed) and rerun — all three stdout byte-match the doc verbatim blocks.
- `housing.csv` — California Housing data file. `RULES.md` — this file. `Exp4_Viva.md` — viva Q&A for Experiment 4.
- Git commits: short and funny messages.
