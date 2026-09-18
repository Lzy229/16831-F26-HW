# record the standard derivation of two dagger results

import matplotlib.pyplot as plt
import numpy as np

steps = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

# hopper
eval_std = np.array([603.3, 259.3, 8.026, 3.56, 2.554, 2.992, 4.554, 3.477, 3.136, 0.8943])
eval_means = np.array([1836, 1554, 3703, 3717, 3720, 3716, 3719, 3738, 3720, 3716])
train_std = [1.948, 3.475, 47.16, 0, 0, 0, 0, 0, 0, 0]

# ant
# eval_std = np.array([98.29, 76.99, 30.39, 1482, 803.9, 228.9, 83.81, 67.2, 118.9, 108.6])
# eval_means = np.array([4758, 4713, 4681, 4081, 4314, 4703, 4703, 4742, 4638, 4733])
# train_std = [12.2, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# Create the plot
plt.errorbar(
    steps, eval_means, 
    yerr=eval_std,      # Pass the errors for the y-axis
    fmt='o',            # 'o' means circle markers; use 'none' to hide connecting lines
    ecolor='red',       # Color of the error bars
    elinewidth=2,       # Thickness of the error bar lines
    capsize=4,          # Length of the horizontal caps at the ends
    capthick=2          # Thickness of the caps
)

plt.xlabel('Steps')
plt.ylabel('Return')
plt.title('Line Plot with Error Bars')
plt.legend()
plt.show()