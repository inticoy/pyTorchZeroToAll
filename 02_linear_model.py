import numpy as np
import matplotlib.pyplot as plt

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]


# our model for the forward pass
def forward(x):
    return x * w


# Loss function
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) * (y_pred - y)


# List of weights/Mean square Error (Mse) for each input
weight_list = []
mse_list = []

# 0.0, 0.1, 0.2 ... 4.0
for w in np.arange(0.0, 4.1, 0.1):
    # Print the weights and initialize the lost
    print("w = %.1f" %w)
    loss_sum = 0

    for x_val, y_val in zip(x_data, y_data):
        # For each input and output, calculate y_hat
        # Compute the total loss and add to the total error
        y_pred_val = forward(x_val)
        l = loss(x_val, y_val)
        loss_sum += l
        print("\t%.1f\t%.1f\t%.1f\t%.1f" %(x_val, y_val, y_pred_val, l))
    # Now compute the Mean squared error (mse) of each
    # Aggregate the weight/mse from this run
    print("MSE =", loss_sum / len(x_data))
    weight_list.append(w)
    mse_list.append(loss_sum / len(x_data))
    print()

# Plot it all
plt.plot(weight_list, mse_list)
plt.ylabel('Loss')
plt.xlabel('w')
plt.show()