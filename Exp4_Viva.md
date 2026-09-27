# Viva Questions — Experiment 4 (Simple Linear Regression)

Dataset: California Housing. 4A/4B use `housing.csv` with X = TotalRooms, Y = MedHouseVal (in \$100k), 80/20 split (seed 42), 16512 train / 4128 test. 4C uses `fetch_california_housing` with X = AveRooms, 80/20 split (`random_state=42`).

## Title, Objective and Aim

**1. What is the title of the experiment?**
Simple Linear Regression using Least Squares (4A), Gradient Descent (4B), and Scikit-learn (4C).

**2. What is the objective?**
To implement simple linear regression on housing data by least squares and gradient descent (X = TotalRooms, Y = MedHouseVal), and evaluate it with MSE, R-squared and the regression-line plot.

**3. What is the aim?**
To fit $\hat{Y} = mX + b$, predict test values, and assess the fit with MSE and $R^2$.

## 4A — Least Squares

**4. What is the hypothesis (regression equation)?**
$\hat{Y} = mX + b$, where $m$ is the slope and $b$ the intercept.

**5. How are slope and intercept computed in least squares?**
$$m = \frac{\sum_{i=1}^{n}(X_i-\bar{X})(Y_i-\bar{Y})}{\sum_{i=1}^{n}(X_i-\bar{X})^2} \qquad b = \bar{Y} - m\bar{X}$$

**6. What values did you get?**
$m = 0.000072$, $b = 1.8840$, so $\hat{Y} = 0.000072X + 1.8840$. Test MSE $= 1.3236$, $R^2 = 0.0167$.

**7. What is MSE?**
$$\mathrm{MSE} = \frac{1}{n}\sum_{i=1}^{n}(Y_i-\hat{Y}_i)^2$$
Average squared prediction error; lower is better.

**8. What is $R^2$?**
$$R^2 = 1 - \frac{\sum_{i=1}^{n}(Y_i-\hat{Y}_i)^2}{\sum_{i=1}^{n}(Y_i-\bar{Y})^2}$$
Fraction of target variance explained. $R^2 = 0.0167$ means TotalRooms alone explains almost nothing — the model underfits.

**9. What is the normal (standard) equation?**
$$\theta = (X^TX)^{-1}X^Ty, \qquad X = [\mathbf{1}\;X],\ \theta = [b, m]^T$$
Closed-form least-squares solution, no iteration needed.

## 4B — Gradient Descent

**10. What is the cost function?**
$$J(m, c) = \frac{1}{n}\sum_{i=1}^{n}(Y_i-\hat{Y}_i)^2$$
Same as MSE. Goal: find $m, c$ that minimize it.

**11. Write the gradients.**
$$\frac{\partial J}{\partial m} = -\frac{2}{n}\sum_{i=1}^{n}X_i(Y_i-\hat{Y}_i) \qquad \frac{\partial J}{\partial b} = -\frac{2}{n}\sum_{i=1}^{n}(Y_i-\hat{Y}_i)$$

**12. Write the update rules.**
$$m := m - \alpha\frac{\partial J}{\partial m} \qquad b := b - \alpha\frac{\partial J}{\partial b}$$
Move opposite to the gradient by step size $\alpha$ (here $\alpha = 0.01$).

**13. Why standardize the feature first?**
Gradient descent converges slowly or diverges when features have large values (TotalRooms goes up to 39320). Standardizing ($z = (x-\mu)/\sigma$) keeps gradients stable. Mean and std must come from the **training** data only.

**14. What were your hyperparameters and result?**
$m = c = 0$ init, $\alpha = 0.01$, 100 epochs. Final: $m = 0.000062$, $b = 1.6341$, test MSE $= 1.3885$, $R^2 = -0.0315$. Cost fell $5.6208 \to 1.3827$ but had not converged — more epochs (or larger $\alpha$) would approach the least-squares solution.

**15. What does a negative $R^2$ mean?**
The model fits worse than predicting the mean $\bar{Y}$ every time — here only because gradient descent stopped early, not because the line is wrong.

**16. How do you verify convergence?**
Plot cost vs epochs — the curve must flatten. Ours was still falling at epoch 100.

## 4C — Scikit-learn

**17. Which feature/target and split did you use?**
X = AveRooms, Y = MedHouseVal, `train_test_split(..., test_size=0.2, random_state=42)`.

**18. How do you train and read parameters?**
```python
model = LinearRegression()
model.fit(X_train, y_train)
```
Slope $m$ = `model.coef_[0]` $= 0.076756$, intercept $b$ = `model.intercept_` $= 1.6548$.

**19. How do you evaluate?**
`mean_squared_error` and `r2_score` on the test set: MSE $= 1.2923$, $R^2 = 0.0138$.

**20. Least squares vs gradient descent vs sklearn — how do they relate?**
All three fit the same line $\hat{Y} = mX + b$. Least squares and sklearn's `LinearRegression` give the exact closed-form answer; gradient descent approaches it iteratively. On the same data and split, least squares and a fully converged gradient descent agree exactly.

**21. Why is $R^2$ near zero in all three?**
One room-count feature cannot explain house prices (location, income, etc. matter). Single-feature linear regression underfits here; that is expected and is visible in the scattered plot with a nearly flat fitted line.

## Dataset

**22. What dataset is used in 4A/4B?**
`housing.csv` (California Housing): 20640 usable rows, 0 missing. X = TotalRooms, Y = MedHouseVal (in \$100k). Train 16512 / test 4128 with 80/20 split (seed 42). 4C uses `fetch_california_housing` (20640 samples, 8 features) with X = AveRooms.

## Procedure

**23. Recite the 4A (least squares) procedure.**
B) Load the dataset. C) Identify X = TotalRooms, Y = MedHouseVal. D) Inspect for missing/duplicate/invalid values. E) Compute slope $m$ and intercept $b$ by least squares. F) Form $\hat{Y} = mX + b$. G) Predict the test set. H) Report MSE and $R^2$. I) Plot points with the regression line.

**24. Recite the 4B (gradient descent) procedure.**
Load and clean the data; 80/20 split; standardize X with train mean/std; init $m = c = 0$; set $\alpha = 0.01$, 100 epochs; loop predict $\rightarrow$ cost $J$ $\rightarrow$ gradients $\rightarrow$ update $m, b$; plot cost vs epochs; predict test data; report MSE and $R^2$.

**25. Recite the 4C (Scikit-learn) procedure.**
Import libraries; fetch the dataset; build the DataFrame and explore it; take X = AveRooms, Y = MedHouseVal; scatter plot; 80/20 split (`random_state=42`); fit `LinearRegression`; read $m, b$; predict; report MSE and $R^2$; plot with labels, title and legend.

## Program

**26. Which libraries does each program use and why?**
4A/4B: `csv` (load file) + NumPy (arrays, statistics, linear algebra) + Matplotlib (plots). 4C adds Pandas (DataFrame) and sklearn (`model_selection`, `linear_model`, `metrics`).

**27. Which lines do the train/test split?**
4A/4B: shuffle indices with seed 42 and slice 80/20. 4C: `train_test_split(X, y, test_size=0.2, random_state=42)`.

**28. Which lines compute the model?**
4A: the $m, b$ least-squares formulas. 4B: the 100-epoch loop with gradient updates. 4C: `model.fit(X_train, y_train)`.

## Output

**29. Read out the three outputs.**
4A: $m = 0.000072$, $b = 1.8840$, MSE $= 1.3236$, $R^2 = 0.0167$. 4B: $m = 0.000062$, $b = 1.6341$, cost $5.6208 \to 1.3827$, MSE $= 1.3885$. 4C: $m = 0.076756$, $b = 1.6548$, MSE $= 1.2923$, $R^2 = 0.0138$.

## Result

**30. State the conclusion of the experiment.**
The programs ran successfully and outputs were verified. Least squares and sklearn give the exact line; gradient descent approaches it but needed more than 100 epochs. One room-count feature underfits house prices ($R^2 \approx 0$), which the flat fitted lines confirm.
