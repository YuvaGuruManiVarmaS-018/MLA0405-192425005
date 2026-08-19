import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))   # <-- indented properly

# Generate values from -5 to 5 with step 0.1
z = np.arange(-5, 5, 0.1)
plt.plot(z, sigmoid(z), color='pink')
plt.title('Visualization of the Sigmoid Function')
plt.xlabel('z')
plt.ylabel('sigmoid(z)')
plt.grid(True)
plt.show()
