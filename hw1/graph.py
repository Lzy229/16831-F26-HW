import matplotlib.pyplot as plt
import numpy as np

steps = np.array([1000, 1500, 2000, 2500, 3000, 3500, 4000,
                  4500, 5000, 5500, 6000, 6500, 7000, 7500,
                  8000, 8500, 9000, 9500, 10000])

means = np.array([1108, 841, 885, 1141, 1245, 1258, 
                  1388, 1432, 1216, 1329, 1311, 925,
                  1836, 1303, 1589, 1413, 1076, 1505, 1433])

plt.plot(steps, means)

# naming the x axis
plt.xlabel('num_agent_train_steps_per_iter')
# naming the y axis
plt.ylabel('Eval_AverageReturn')

plt.show()