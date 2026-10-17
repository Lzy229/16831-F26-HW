import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

files = glob.glob("data/experiment4/q4_search_*_lr0.01*.csv")

plt.figure(figsize=(10, 6))

line_styles = ['--', '-.', ':']

for i, file in enumerate(files):
    df = pd.read_csv(file)

    # Suppose your CSV has columns: x and y
    plt.plot(df["Step"], df["Value"], linestyle=line_styles[i % len(line_styles)],label=os.path.basename(file[:-24]))

plt.xlabel("Step")
plt.ylabel("AverageReturn")
plt.title("q4_batch_size")
plt.legend(fontsize=8)
plt.grid(True)

plt.show()