# ML Lab Record — Rules & Constraints

Running rules for the PCCSL 507 Machine Learning lab record (LaTeX). Follow these on every experiment unless the user overrides them.

## 1. Document setup (from the .docx template, keep same structure)
- Paper: US Letter. Margins: top 0.7in, bottom 0.63in, left/right 0.75in.
- Font: Times clone (`mathptmx`), 11pt base. Section headings: bold 14pt, **left-aligned** (template style). No `Title of the experiment:` prefix — bare title text only. Never use short forms SLR/MLR — always Simple/Multiple Linear Regression.
- Colors: primary `#1F3864`, grey text `#555555`, table grid `#999999`.
- Header (all pages): `PCCSL 507 Machine Learning Laboratory Record`, bold primary, primary bottom rule.
- Footer (all pages): `DEPARTMENT OF CSE ... CAPE COLLEGE OF ENGINEERING ALAPPUZHA ... Page No. <n>`.
- Cover institution line: `CAPE COLLEGE OF ENGINEERING ALAPPUZHA`.
- Cover details (Name … Faculty In-Charge, Signature/Date): centered block, fixed-width label column, vertically aligned colons and write-in lines.
- Index table: header row dark-blue fill + white bold text; 15 numbered rows; fill in each experiment's title + page number.

## 2. Code rules (strict)
- **No scikit-learn anywhere except Experiment 4C** (no `fetch_*`, no `train_test_split`, no metrics elsewhere). NumPy + stdlib only; `matplotlib` allowed for plots.
- 4C exception: uses `fetch_california_housing` + `LinearRegression` (X=AveRooms, 80/20 `random_state=42`).
- Data always comes from the **`housing.csv` file** (California Housing), never a library loader.
- Experiment 1 fixed spec: X = TotalRooms, Y = MedHouseVal (in $100k), 80/20 split with seed 42, standardize with train mean/std.
- Least squares via classic formulas; gradient descent with m = c = 0, α = 0.01, 100 epochs, MSE cost `J = 1/n Σ(Y−Ŷ)²`, gradients with −2/n factor.
- Metrics: MSE, R² (and RMSE). Real run output only — never invent numbers.
- No `#` comments in any program or listing. Listings must match their `.py` file exactly.
- Code style: simple, beginner-readable, faculty notation (`m`, `b`), plain statements with no comments.

## 3. Section content rules (Experiment 1 pattern)
- TITLE: short, filled in. OBJECTIVE: filled in (2–3 lines). AIM: short (1–2 lines).
- THEORY: equations only (regression equation, LS estimates, MSE cost, gradients, updates, MSE/R²).
- PROCEDURE: faculty style — Part A least squares steps + Part B gradient-descent steps.
- DATASET DESCRIPTION: real counts (usable/missing/duplicates/invalid, train/test sizes, feature, target).
- PROGRAM: full listing, must match its `.py` file exactly. OUTPUT: real console output + plots (sales exp has bar + pie side by side). RESULT: always the exact template sentence with the model-name blank filled in **bold, no underline**.
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
- Joined order: 1A marks, 1B dataframe, 2 sales (+bar+pie charts), 3A hours-score, 3B MLR, 4A LS, 4B GD, 4C sklearn. Index page numbers must match section starts; verify with `pdftotext` after every compile.
- `ML_Lab_Experiment1.tex` / `.pdf` — full record: cover, index, experiment.
- `ML_Lab_Report_Template.tex` — blank template (program/output pages before experiment sheet).
- `exp1_run.py` — the exact program in the listing. `exp1_ls.png`, `exp1_cost.png` — the figures.
- `ML_Lab_Experiment2.tex` / `.pdf` — Experiment 2 (NumPy marks array), same template. `exp2_run.py` — its program.
- `ML_Lab_Experiment3.tex` / `.pdf` — Experiment 3 (pandas student dataframe + CSV). `exp3_run.py`, `exp3_students.csv`.
- `ML_Lab_Experiment4.tex` / `.pdf` — Experiment 4 (NumPy 2D sales array). `exp4_run.py`.
- `ML_Lab_Experiment5.tex` / `.pdf` — Experiment 5 (SLR hours vs score, predict at 9). `exp5_run.py`, `exp5_fit.png`.
- `ML_Lab_Experiment6.tex` / `.pdf` — Experiment 6 (MLR area+age, 3D plane). `exp6_run.py`, `exp6_plane.png`.
- Pandas is allowed where the experiment needs DataFrames (Exp 3); ML model code stays NumPy-only.
- `housing.csv` — California Housing data file. `RULES.md` — this file. `Exp4_Viva.md` — viva Q&A for Experiment 4 with equations.
