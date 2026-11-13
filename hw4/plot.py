import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import numpy as np

# for experiment2
# steps = np.array([0])

# Eval_AverageReturn = np.array([-61.01])
# Train_AverageReturn = np.array([-172.3])

# # plt.plot(steps, Eval_AverageReturn, label="Eval_AverageReturn", color="blue")
# # plt.plot(steps, Train_AverageReturn, label="Train_AverageReturn", color="red")

# plt.scatter(steps, Eval_AverageReturn, c='red', label="Eval_AverageReturn")
# plt.scatter(steps, Train_AverageReturn, c='blue', label="Train_AverageReturn")

# plt.xlabel("Learning Iter")
# plt.ylabel("Mean Return")
# plt.title("Q2 results")

# plt.legend()

# plt.show()

# for experiment3
# files = glob.glob("./rob831/data/experiments3/*obsta*.csv")

# plt.figure(figsize=(10, 6))

# line_styles = ['--', '-.', ':']

# steps = []
# EvalReturn = []

# for i, file in enumerate(files):
#     df = pd.read_csv(file)

#     # Suppose your CSV has columns: x and y
#     # plt.plot(df["Step"], df["Value"], linestyle=line_styles[i % len(line_styles)],label=os.path.basename(file[:-24]))
#     steps = (df["Step"])
#     EvalReturn = (df["Value"])


# # steps = np.array(steps)
# # values = np.array(EvalReturn)

# plt.plot(steps, EvalReturn, color='blue', label="Eval Mean", linewidth=2)

# plt.xlabel("Learning Iter")
# plt.ylabel("Eval MeanReturn")
# plt.title("Q3 obstacles")
# plt.grid(True)

# plt.legend()

# plt.show()

# for experiment4 
files = glob.glob("./rob831/data/experiments4/*ensemble*.csv")

plt.figure(figsize=(10, 6))

line_styles = ['--', '-.', ':']

for i, file in enumerate(files):
    df = pd.read_csv(file)

    # Suppose your CSV has columns: x and y
    plt.plot(df["Step"], df["Value"], linestyle=line_styles[i % len(line_styles)],label=os.path.basename(file[:-24]))

plt.xlabel("Step")
plt.ylabel("AverageReturn")
plt.title("q4_ensemble")
plt.legend(fontsize=8)
plt.grid(True)

plt.show()