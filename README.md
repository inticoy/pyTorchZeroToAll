# PyTorchZeroToAll - Sung Kim

# Lecture 00 : Info

- [https://www.youtube.com/playlist?list=PLZSPcxXPEZVqrk1CYgM2yJ4BPXxiGSyIp](https://www.youtube.com/playlist?list=PLZSPcxXPEZVqrk1CYgM2yJ4BPXxiGSyIp)
- [https://github.com/hunkim/PyTorchZeroToAll](https://github.com/hunkim/PyTorchZeroToAll)

# Lecture 01 : Overview

## What is Human Intelligence?

- 정보의 유입과 그것을 바탕으로 결정하는 것
- 머신러닝 역시 마찬가지

## Machine Learning

- (Labeled) Dataset을 바탕으로 학습
- AI → Machine Learning → Representation Learning → Deep Learning (가장 세분화)

## Why PyTorch?

- More Pythonic
- More Neural Networkic

## Check PyTorch Version

```python
import torch
print(torch.__version__)
```

## Topics

- Linear, Logistic, softmax models
- DNN
- CNN
- RNN
- 기타 등등

# Lecture 02 : Linear Model

## Model Design

- $\hat y = x \times m + b$

## Linear Regression (선형 회귀)

- 기계는 m과 b를 random하게 추측하게 됨.

## Traning Loss (Error)

- $loss = (\hat y - y)^2$
- $MSE = \frac 1 N \sum\limits_{n=1}^N (\hat y_n - y_n)^2$
    - Mean Square Error

# Lecture 03 : Gradient Descent

- Loss 를 최소화 할 수 있는 w를 찾자

## Gradient Descent Algorithm

![Untitled](PyTorch%20Lecture%20-%20Sung%20Kim%20050e2b367db54ab9a3567f4fa90bfa9c/Untitled.png)

- Lecture 02에서 그린 Loss Graph의 기울기를 바탕으로 w의 값을 줄일지 늘릴지 결정함.
- 즉 기울기가 0이 되도록 w값을 조정하는 알고리즘
- 기울기가 음수로 크면 w값을 그에 비례하여 키우는 식

## Mathematical Approach

- $loss = (\hat y - y)^2 = (x \times w - y)^2$
- $Gradient = \frac {\delta loss} {\delta w}$
- $w = w - \alpha \frac {\delta loss} {\delta w} = w - 2\alpha x(xw - y)$
    - $- \alpha$인 이유는 기울기가 음수일 때는 w값을 키워야 하므로
- $\alpha$  : Learning Rate (학습률, 대체로 0.01과 같이 매우 작은 값)

# Lecture 04 : Back-propagation and Autograd

# Lecture 05 : Linear Regression in the PyTorch

# Lecture 06 : Logistic Regression

# Lecture 07 : Wide and Deep

# Lecture 08 : PyTorch DataLoader

# Lecture 09 : Softmax Classfier