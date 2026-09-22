# Viva Questions — Experiment 4 (Simple Linear Regression)

Dataset: California Housing. 4A/4B use `housing.csv` with X = TotalRooms, Y = MedHouseVal (in \$100k), 80/20 split (seed 42), 16512 train / 4128 test. 4C uses `fetch_california_housing` with X = AveRooms, 80/20 split (`random_state=42`).

## 4A — Least Squares

**1. What is the hypothesis (regression equation)?**
$\hat{Y} = mX + b$, where $m$ is the slope and $b$ the intercept.

**2. How are slope and intercept computed in least squares?**
$$m = \frac{\sum_{i=1}^{n}(X_i-\bar{X})(Y_i-\bar{Y})}{\sum_{i=1}^{n}(X_i-\bar{X})^2} \qquad b = \bar{Y} - m\bar{X}$$

**3. What values did you get?**
$m = 0.000072$, $b = 1.8840$, so $\hat{Y} = 0.000072X + 1.8840$. Test MSE $= 1.3236$, $R^2 = 0.0167$.

**4. What is MSE?**
$$\mathrm{MSE} = \frac{1}{n}\sum_{i=1}^{n}(Y_i-\hat{Y}_i)^2$$
Average squared prediction error; lower is better.

**5. What is $R^2$?**
$$R^2 = 1 - \frac{\sum_{i=1}^{n}(Y_i-\hat{Y}_i)^2}{\sum_{i=1}^{n}(Y_i-\bar{Y})^2}$$
Fraction of target variance explained. $R^2 = 0.0167$ means TotalRooms alone explains almost nothing — the model underfits.

**6. What is the normal (standard) equation?**
$$\theta = (X^TX)^{-1}X^Ty, \qquad X = [\mathbf{1}\;X],\ \theta = [b, m]^T$$
Closed-form least-squares solution, no iteration needed.

## 4B — Gradient Descent

**7. What is the cost function?**
$$J(m, c) = \frac{1}{n}\sum_{i=1}^{n}(Y_i-\hat{Y}_i)^2$$
Same as MSE. Goal: find $m, c$ that minimize it.

**8. Write the gradients.**
$$\frac{\partial J}{\partial m} = -\frac{2}{n}\sum_{i=1}^{n}X_i(Y_i-\hat{Y}_i) \qquad \frac{\partial J}{\partial b} = -\frac{2}{n}\sum_{i=1}^{n}(Y_i-\hat{Y}_i)$$

**9. Write the update rules.**
$$m := m - \alpha\frac{\partial J}{\partial m} \qquad b := b - \alpha\frac{\partial J}{\partial b}$$
Move opposite to the gradient by step size $\alpha$ (here $\alpha = 0.01$).

**10. Why standardize the feature first?**
Gradient descent converges slowly or diverges when features have large values (TotalRooms goes up to 39320). Standardizing ($z = (x-\mu)/\sigma$) keeps gradients stable. Mean and std must come from the **training** data only.

**11. What were your hyperparameters and result?**
$m = c = 0$ init, $\alpha = 0.01$, 100 epochs. Final: $m = 0.000062$, $b = 1.6341$, test MSE $= 1.3885$, $R^2 = -0.0315$. Cost fell $5.6208 \to 1.3827$ but had not converged — more epochs (or larger $\alpha$) would approach the least-squares solution.

**12. What does a negative $R^2$ mean?**
The model fits worse than predicting the mean $\bar{Y}$ every time — here only because gradient descent stopped early, not because the line is wrong.

**13. How do you verify convergence?**
Plot cost vs epochs — the curve must flatten. Ours was still falling at epoch 100.

## 4C — Scikit-learn

**14. Which feature/target and split did you use?**
X = AveRooms, Y = MedHouseVal, `train_test_split(..., test_size=0.2, random_state=42)`.

**15. How do you train and read parameters?**
```python
model = LinearRegression()
model.fit(X_train, y_train)
```
Slope $m$ = `model.coef_[0]` $= 0.076756$, intercept $b$ = `model.intercept_` $= 1.6548$.

**16. How do you evaluate?**
`mean_squared_error` and `r2_score` on the test set: MSE $= 1.2923$, $R^2 = 0.0138$.

**17. Least squares vs gradient descent vs sklearn — how do they relate?**
All three fit the same line $\hat{Y} = mX + b$. Least squares and sklearn's `LinearRegression` give the exact closed-form answer; gradient descent approaches it iteratively. On the same data and split, least squares and a fully converged gradient descent agree exactly.

**18. Why is $R^2$ near zero in all three?**
One room-count feature cannot explain house prices (location, income, etc. matter). Single-feature linear regression underfits here; that is expected and is visible in the scattered plot with a nearly flat fitted line.
