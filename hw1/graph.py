import matplotlib.pyplot as plt
import numpy as np

# the graph of small experiments
# steps = np.array([1000, 1500, 2000, 2500, 3000, 3500, 4000,
#                   4500, 5000, 5500, 6000, 6500, 7000, 7500,
#                   8000, 8500, 9000, 9500, 10000])

# means = np.array([1108, 841, 885, 1141, 1245, 1258, 
#                   1388, 1432, 1216, 1329, 1311, 925,
#                   1836, 1303, 1589, 1413, 1076, 1505, 1433])

# plt.plot(steps, means)

# # naming the x axis
# plt.xlabel('num_agent_train_steps_per_iter')
# # naming the y axis
# plt.ylabel('Eval_AverageReturn')

# plt.show()

# the graph of Ant 
# steps = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

# means_of_bc = np.array([4758, 4713, 4681, 4081, 4314, 4703, 4703, 4742, 4638, 4733])
# means_of_expert = np.array([4714, 4669, 4883, 4708, 4695, 4947, 4912, 4707, 4641, 4658])

# plt.plot(steps, means_of_bc, label="BC agent", color="blue")
# plt.plot(steps, means_of_expert, label="expert", color="red")

# plt.xlabel("Learning Iter")
# plt.ylabel("Mean Return")
# plt.legend()

# plt.show()


# the graph of Hopper
steps = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

means_of_bc = np.array([1836, 1554, 3703, 3717, 3720, 3716, 3719, 3738, 3720, 3716])
means_of_expert = np.array([3773, 1809, 1555, 3688, 3721, 3717, 3715, 3714, 3739, 3719])

plt.plot(steps, means_of_bc, label="BC agent", color="blue")
plt.plot(steps, means_of_expert, label="expert", color="red")

plt.xlabel("Learning Iter")
plt.ylabel("Mean Return")
plt.legend()

plt.show()