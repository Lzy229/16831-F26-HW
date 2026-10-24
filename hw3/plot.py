import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import numpy as np

files_dqn = glob.glob("data/experiment1/train_bestreturn/q1_dqn*.csv")
files_ddqn = glob.glob("data/experiment1/train_bestreturn/q1_doubledqn*.csv")

plt.figure(figsize=(10, 6))

line_styles = ['--', '-.', ':']

df_Step_dqn = []
df_Values_dqn = []

for i, file in enumerate(files_dqn):
    df = pd.read_csv(file)

    # Suppose your CSV has columns: x and y
    # plt.plot(df["Step"], df["Value"], linestyle=line_styles[i % len(line_styles)],label=os.path.basename(file[:-24]))
    df_Step_dqn.append(df["Step"])
    df_Values_dqn.append(df["Value"])


print(df_Step_dqn)
steps = np.array(df_Step_dqn)
values = np.array(df_Values_dqn)

# Average and standard deviation across the 3 runs
mean_values = np.mean(values, axis=0)
std_values = np.std(values, axis=0, ddof=1)

dqn_color = "tab:blue"

line, =plt.plot(steps[0], mean_values, color=dqn_color, label="DQN Mean", linewidth=2)

# dqn_color = line.get_color()

plt.errorbar(
    steps[0], mean_values, 
    yerr=std_values,      # Pass the errors for the y-axis
    # fmt='o',            # 'o' means circle markers; use 'none' to hide connecting lines
    color=dqn_color,       # Color of the error bars
    elinewidth=2,       # Thickness of the error bar lines
    capsize=4,          # Length of the horizontal caps at the ends
    capthick=2          # Thickness of the caps
)

df_Step_ddqn = []
df_Values_ddqn = []

for i, file in enumerate(files_ddqn):
    df = pd.read_csv(file)

    # Suppose your CSV has columns: x and y
    # plt.plot(df["Step"], df["Value"], linestyle=line_styles[i % len(line_styles)],label=os.path.basename(file[:-24]))
    df_Step_ddqn.append(df["Step"])
    df_Values_ddqn.append(df["Value"])

steps = np.array(df_Step_ddqn)
values = np.array(df_Values_ddqn)

# Average and standard deviation across the 3 runs
mean_values = np.mean(values, axis=0)
std_values = np.std(values, axis=0, ddof=1)

ddqn_color = "tab:orange"

line, = plt.plot(steps[0], mean_values, label="DDQN Mean", color=ddqn_color, linewidth=2)

plt.errorbar(
    steps[0], mean_values, 
    yerr=std_values,      # Pass the errors for the y-axis
    # fmt='o',            # 'o' means circle markers; use 'none' to hide connecting lines
    color=ddqn_color,       # Color of the error bars
    elinewidth=2,       # Thickness of the error bar lines
    capsize=4,          # Length of the horizontal caps at the ends
    capthick=2          # Thickness of the caps
)

plt.xlabel("Step")
plt.ylabel("Train_BestReturn")
plt.title("DQN & DDQN best return")
plt.legend(fontsize=8)
plt.grid(True)

plt.show()