import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1/(1+np.exp(-x))

x = np.arange(-5.0, 5.0, 1)
y = sigmoid(x)

plt.plot(x, y, ':b')
plt.plot([0,0], [1.0, 0.0], ":")
# plt.axis([-5, 5, 0, 1])
plt.show()