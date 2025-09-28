---
title: "Predicting Bitcoin Prices with Linear Regression: A Beginner-Friendly Guide"
description: "Beginner's guide using linear regression to predict Bitcoin prices. Learn key concepts, model building, and testing with historical data"
tags: ['ml', 'Machine Learning', 'AI', 'Artificial Intelligence', 'Linear Regression']
date: 2025-09-19
params:
  math: true
link: https://blog.pvcodes.in/bitcoin-price-prediction-linear-regression
# haslastmod: true
---

Linear Regression is one of the most fundamental concepts in machine learning and an excellent starting point for beginners. In this blog, we’ll unpack what linear regression is, how it works, and then build a practical model to predict Bitcoin closing prices using **historical BTC price data**.

*Note: The python notebook is been attached, feel free to try it yourself.*

---

## What is Linear Regression?

Linear regression is a simple way to understand how one thing affects another. Imagine trying to predict a student’s exam score based on how many hours they study. Linear regression fits a straight line through data points, showing the relationship between study hours (independent variable) and exam scores (dependent variable). This helps predict what the score might be for a given number of study hours by using a simple formula like \\(y =  w\cdot x + b\\), where \\(y\\) is the score, \\(x\\) is the hours studied, \\(w\\) is how much the score changes with each hour, and \\(b\\) is the starting point when no hours are studied.

This technique is easy to use and understand, making it popular for many fields like business, healthcare, and tech. It helps you figure out how different factors are connected and predict future outcomes based on past data. Whether you’re using one factor or many, linear regression is a valuable tool to turn data into useful insights quickly and clearly.

More formally, Linear regression models the relationship between one dependent variable \\(y\\) and one or more independent variables \\(x\\). It “fits” a straight line (or a hyperplane for multiple features) that best predicts \\(y\\) from \\(x\\).

Linear regression models are categorized according to the number of input features (\\(x\\)) they use.

1. **Simple Linear Regression (one feature):**

$$y = mx+b$$

where,

* \\(x\\): feature (independent variable)

* \\(y\\): target (dependent variable)

* \\(w\\): slope parameter

* \\(b\\): intercept

![Simple Regression Model](https://cdn.hashnode.com/res/hashnode/image/upload/v1759061232232/511b07c2-954c-4fca-8a2d-40a81ce4e994.jpeg)

2. **Multiple Linear Regression (many features):**

    Multiple Linear Regression models the relationship between a dependent variable and two or more independent variables to understand how multiple factors together influence the outcome. It fits a linear equation that predicts the result based on the combined effects of all input features.

$$y = \vec w \cdot \vec x + b$$

where,

* \\(\vec x\\): vector of input features \\([x_1, x_2, …, x_n]\\)

* \\(\vec w\\): vector of learned weights \\([w_1, w_2, …, w_n]\\)

* \\(b\\): bias (intercept)

* \\(y\\): predicted output

* \\(\vec w \cdot \vec x\\): the [dot product](https://en.wikipedia.org/wiki/Dot_product) of \\(\vec w\\) and \\(\vec x \\)

![Multiple Regression Model](https://cdn.hashnode.com/res/hashnode/image/upload/v1759061337877/2bcb260c-4a36-469a-a487-ba313b319b14.png)

This above plot is the 3d representation of Multiple Linear Regression Model, having `wt`, `mpg` and `year` as the feature and target respectively, This images was sourced from - [stat420.org](https://book.stat420.org/multiple-linear-regression.html).

The objective is to find \\(\vec w\\) and \\(b\\) that minimize the difference between predicted values \\(\hat y\\) and actual values \\(y\\). **So that for any feature variables** (\\(\vec x\\)) **we are able to predict/compute the target** (\\(y\\)).

---

## Measuring Fit: Mean Squared Error (MSE)

Mean Squared Error (MSE) is a common way to measure how well a linear regression model fits the data. It calculates the average of the squared differences between the actual values and the values predicted by the model, with lower MSE values indicating better prediction accuracy.

We evaluate how good the predictions are using **Mean Squared Error (MSE):**

$$J(\mathbf{w}, b) = \frac{1}{2m} \sum_{i=1}^{m} \big(\hat{y}^{(i)} - y^{(i)}\big)^2$$

where,

* \\(m\\) is the number of examples.

* \\(\hat y\\) is the predicted value from the model, \\(\hat y=\vec w \cdot \vec x\\)

Minimizing this gives us the “best-fit” line.

---

## How Does Linear Regression Work?

Linear regression works by finding the best-fitting straight line through a set of data points that relate an independent variable (input) to a dependent variable (output). This line is defined by an equation that predicts the output based on the input, and the model adjusts this line to minimize the difference between the actual and predicted values, often by using a method called least squares. The goal is to create a simple equation that accurately represents the relationship so it can be used to make predictions on new data.

The most common way to fit parameters is with the **Least Squares Method**, solved via **Gradient Descent**.

### Gradient Descent in a Nutshell

Gradient descent is an iterative algorithm that updates parameters step by step in the direction that decreases the cost function the fastest.

The updates are:

$$w = w - \alpha \cdot \frac{\partial J}{\partial w}, \quad b = b - \alpha \cdot \frac{\partial J}{\partial b}$$

* \\(\alpha\\): learning rate (step size)

* \\(\frac{\partial J}{\partial w}, \frac{\partial J}{\partial b}\\): gradients

**The loop continues until the error stabilizes (*convergence*).**

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1759061359714/d7a9e5ad-aaa9-4c66-aa1f-41773b21de47.png)

## Why Vectorization Matters?

In machine learning, datasets can have millions of rows. Loops in Python quickly become inefficient. **Vectorization** leverages NumPy’s optimized operations to handle entire vectors or matrices in a single step, making code faster and cleaner.

* **Without vectorization (slow loop):**

    ```python
    y_hat = []  
        for i in range(m):
         pred = 0
         for j in range(n):  
        pred += w[j] * X[i, j]  
        y_hat.append(pred + b)
    ```

* **With vectorization (fast & concise):**

    ```python
    y_hat = np.dot(X, w) + b
    ```

*Note: Both produce the same result, but vectorization is significantly faster.*

## Case Study: Predicting Bitcoin Closing Prices

We’ll now apply linear regression on Bitcoin’s historical price data (2014–2024). Explore the dataset for yourself [Kaggle BTC-USD Stock Data](https://www.kaggle.com/datasets/gallo33henrique/bitcoin-btc-usd-stock-dataset).

The dataset includes following features:

* **Open** – price at the start of the day

* **High / Low** – daily maximum and minimum prices

* **Close** – target (end-of-day price)

* **Adj Close** – adjusted close; excluded to avoid leakage

* **Volume** – trading volume

### Step 0 - Basic setup for getting started

* Use [Google Colab](https://colab.research.google.com/) or setup the Jupyter Notebook locally ([guide for local setup](https://jupyter.org/install)).

* We are using `Pandas` and `Numpy` libraries.

* Let’s import these libraries

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```

### Step 1 — Load & Inspect the Data

```python
df = pd.read_csv('BTC-USD_stock_data.csv')  
df['Date'] = pd.to_datetime(df['Date'])  
df.set_index('Date', inplace=True)  
df.head()
```

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1759053771161/115ffd52-1cb8-4c35-a9e7-7ee5b6163038.png)

#### Checks Before Modeling

* Ensure numeric dtypes.

* Handle missing values (`df.isna().sum()`).

* Remove duplicates.

Since our dataset is already cleaned, we do not have to worry abot the data sanity

---

### Step 2 — Feature Selection & Normalization

We’ll use: `Open`, `High`, `Low`, `Volume`.  
Exclude `Adj Close` (directly derived from `Close`).

#### Why Normalize?

BTC prices and volume differ enormously in scale. Without scaling, training is inefficient. We apply **z-score normalization**:

$$z = \frac{x - \bar x}{\sigma}$$

where,

* \\(\bar x\\), mean of the \\(x\\) and

* \\(\sigma\\), [standard deviation](https://en.wikipedia.org/wiki/Standard_deviation) of \\(x\\).

```python
df = (df - df.mean()) / df.std(ddof=0)
```

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1759054243352/16203a1a-3fee-471f-bfac-91b43bbde05d.png)

For our model, the `Close` attribute in the dataset is the target (\\(y\\)). Let’s fetch that too.

```python
# Create target variable (next day's closing price)
df['Target'] = df['Close'].shift(-1)
df['Target'] = df['Target'].fillna(df['Target'].mean())
```

### Step 3 — Visualizing Trends

Plot 30-day rolling mean of closing prices:

```python
rolling = df['Target'].rolling(window=30).mean()
plt.figure(figsize=(12,4))  
plt.plot(df.index, df['Target'], alpha=0.3, label='Close')  
plt.plot(df.index, rolling, color='red', label='30-day Rolling Mean')  
plt.title('Close Price with Rolling Mean')  
plt.xlabel('Date')  
plt.ylabel('Close Price (Standardized)')  
plt.legend()  
plt.show()
```

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1759054291029/7ec98290-86f8-482f-b8b8-f707c42965f6.png)

### Step 3 - Extract Feature and Target

```python
# Separate features and target
features = df.loc[:, df.columns != 'Target']
target = df['Target']
print(f'Feature matrix shape: {features.shape}')
print(f'Target vector shape: {target.shape}')
```

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1759054255992/f019a91d-6cfc-412f-8bb1-4732ece013e6.png)

## Step 4 — Building the Model

### 1\. Prediction Function

```python
def model_fn(x, w, b):
    """
    Linear regression prediction function.
    
    Args:
        x: Single Feature Datapoint (1 x n)
        w: Weight vector (n,)
        b: Bias scalar
    
    Returns:
        y_hat: Predicted values (m,)
    """
    y_hat = np.dot(x, w) + b
    return y_hat
```

for example:

\\(\vec x = [-1.127270, -1.12714, -1.126278, -1.203614]\\)

\\(\vec w = [0.18435932, 0.19623844, 0.19707516, 0.20951568, 0.20951568, 0.00232556]\\) (Random coefficients)

\\(b = 0.000397\\)

then,

\\(\hat y = x_1 w_1 + x_2 w_2 + x_3 w_3 + x_5 w_5 + b\\)

\\(\hat y= (-1.127270 \cdot 0.18435932) + (-1.12714 \cdot 0.19623844) + (-1.126278 \cdot 0.19707516) + (-1.203614 \cdot 0.20951568) + 0.000397\\)

or we can get the ***dot product*** of matrix of \\(\vec w\\) and \\(\vec x\\), i.e., \\(\hat y= \vec w \cdot \vec x + b\\)

\\(\vec{x} = \begin{pmatrix} -1.127270 -1.12714 -1.126278 -1.203614 \end{pmatrix} \\)

$$\vec{w} = \begin{pmatrix} 0.18435932 \\ 0.19623844 \\ 0.19707516 \\ 0.20951568 \\ 0.20951568 \\ 0.00232556 \end{pmatrix}$$

\\(\vec{x} \cdot \vec{w} = (-1.127270 \cdot 0.18435932) + (-1.12714 \cdot 0.19623844) + (-1.126278 \cdot 0.19707516) + (-1.203614 \cdot 0.20951568) + 0.000397\\)

### 2\. Cost function

The purpose. is to measure of how bad predictions are using **Mean Squared Error** -

$$J(\mathbf{w}, b) = \frac{1}{2m} \sum_{i=1}^{m} \big(\hat{y}^{(i)} - y^{(i)}\big)^2$$

```python
def cost_fn(X, Y, w, b):
    """
    Calculate Mean Squared Error cost.
    
    Args:
        X: Feature matrix
        Y: Target vector
        w: Weight vector
        b: Bias scalar
    
    Returns:
        cost: Mean squared error
    """
    m = len(X)
    y_hat = model_fn(X, w, b)
    cost = np.sum((y_hat - Y) ** 2) / (2 * m)
    return cost
```

### 3\. Gradient Computation

If \\(\hat y= \vec w \cdot \vec x +b\\), and cost \\(J = \frac {1}{2m} \sum (\hat y - y)^2\\). then,

* Residual vector: \\(r=\hat y − y = \vec w \cdot \vec x + b − y\\), (shape: m,)

* Gradient w.r.t. weights \\(\vec w\\)

    $$\frac{\partial J}{\partial w} = \frac{1}{m} X^\top r$$

* Gradient w.r.t bias \\(b\\)

    $$\frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^m r_i$$

```python
def compute_gradient(X, Y, w, b):
    """
    Compute gradients for weights and bias.
    
    Args:
        X: Feature matrix
        Y: Target vector
        w: Weight vector
        b: Bias scalar
    
    Returns:
        dl_dw: Gradient with respect to weights
        dl_db: Gradient with respect to bias
    """
    m = len(X)
    y_hat = model_fn(X, w, b)
    
    # Compute gradients
    dl_dw = np.dot(X.T, (y_hat - Y)) / m
    dl_db = np.sum(y_hat - Y) / m
    
    return dl_dw, dl_db
```

### 4\. Gradient Descent Algorithm

**Gradient descent core idea**

* Repeatedly move `w` and `b` in the direction that reduces the cost:

    $$w \leftarrow w - \alpha \cdot dl_{dw}, \quad b \leftarrow b - \alpha \cdot dl_{db}$$

* \\(\alpha\\)(alpha), the learning rate controls step size:

  * Too large → divergence (cost blows up).

  * Too small → very slow convergence.

* **Practical training notes**

  * **Initialization**: zeros are fine for linear regression.

  * **Iterations**: monitor cost decrease; don’t blindly run 10k — stop when cost plateaus.

  * **Monitoring**: store `cost_history` and plot it. Also look at the magnitude of gradients — if gradients are near zero early, learning rate may be too small; if gradients explode, rate is too large.

  * **Early stopping**: stop if validation cost increases (overfitting) or if cost changes less than a small epsilon for many steps.

## Step4: Start the model training

```python
# Convert to numpy arrays for efficient computation
X_train = features.to_numpy()
y_train = target.to_numpy()
print(X_train[0].shape)
# Initialize parameters
w_init = np.zeros_like(X_train[0])
b_init = 0.0

# Hyperparameters
alpha = 0.003      # Learning rate
iterations = 10000

# Train the model
print("Training Linear Regression Model...")
w_final, b_final, cost_history = gradient_descent(
    X_train, y_train, w_init, b_init, alpha, iterations
)

print(f"\nFinal parameters:")
print(f"Weights: {w_final}")
print(f"Bias: {b_final:.6f}")
print(f"Final cost: {cost_history[-1]:.6f}")
```

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1759060969201/57574cf0-c165-4b87-ae2b-54d0195a8259.png)

### Learning curve subplots (first 30 / 100 / 1000 iterations)

* **First 30 iterations:** you may see a tiny initial change — this is the model finding an initial descent direction.

* **First 100 iterations:** if the curve bends downward noticeably, gradient descent is making meaningful progress; verify cost is decreasing smoothly.

* **First 1000 iterations:** if this has flattened, the model is converging. If it’s still noisy or increasing, reduce the learning rate. **Tip:** plot `cost_history` on a log scale if values span orders of magnitude — that often makes convergence behavior easier to read.

    ![](https://cdn.hashnode.com/res/hashnode/image/upload/v1759060979196/ad956d8d-cf51-4d76-b853-eb30391ed68b.png)

### Prediction vs Actual — two-panel explanation

The below plot show scatter plot that compares the actual values against the predicted values generated by the linear regression model. Points clustering closely along the diagonal red dashed line indicate strong agreement between predicted and real values

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1759065137576/fafbdd8c-34ce-411a-bed8-3f2f0b03cd03.png)

The second panel presents a time series plot showing how the predicted and actual values evolve over time. By plotting standardized prices on the same timeline, it demonstrates how well the model tracks real-world changes, making it easier to observe periods of strong prediction and potential deviations. Together, these visualizations provide a comprehensive view of the model’s performance both in terms of point-by-point accuracy and temporal consistency.

---

### Link to the notebook

* Jupyter notebook - [pvcodes/bitcoin-price-prediction](https://colab.research.google.com/github/pvcodes/ml/blob/main/btc_price_prediction.ipynb)

* HTML Version - [notes.pvcodes.in/btc-price-prediction-model-multiple-linear-regression](https://notes.pvcodes.in/btc-price-prediction-model-multiple-linear-regression/)
