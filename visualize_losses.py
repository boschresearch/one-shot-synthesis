"""
Go over all experiments in the folder with checkpoints
and visualize losses & logits in png format.
"""


import os
import argparse
import numpy as np
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='checkpoints')
parser.add_argument('--step_x', type=int, default=50)
args = parser.parse_args()

# --- losses ---#
exp_list = os.listdir(args.path)
for item in exp_list:
    cur_dict, cur_plot = dict(), dict()
    try:
        with open(os.path.join(args.path, item, "losses", "losses.csv"), "r") as f:
            cur_file = f.readlines()
    except:
        continue
    for line in cur_file:
        elements = line.replace("\n", "").split(",")
        cur_dict[elements[0]] = np.array(list(map(float, elements[1:])))
    for loss in cur_dict:
        cur_plot[loss.split("__")[0]] = cur_plot.get(loss.split("__")[0], 0) + cur_dict[loss]
    for loss in cur_plot:
        x = np.linspace(0, args.step_x * len(cur_plot[loss]), len(cur_plot[loss]))
        plt.plot(x,  cur_plot[loss], label=loss)
    plt.legend()
    plt.grid(b=True, which='major', color='#666666', linestyle='--')
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='#999999', linestyle='--', alpha=0.2)
    plt.savefig(os.path.join(args.path, item, "losses", "losses.png"))
    plt.close()

# --- logits ---#
colors = {"Dreal.1": 'b', "Dreal.5": 'b', "Dreal.9": 'b',
          "Dfake.1": 'orange', "Dfake.5": 'orange', "Dfake.9": 'orange'}
for item in exp_list:
    fix, ax = plt.subplots(1)
    cur_dict, cur_plot = dict(), dict()
    try:
        with open(os.path.join(args.path, item, "losses", "logits.csv"), "r") as f:
            cur_file = f.readlines()
    except:
        continue
    for line in cur_file:
        elements = line.replace("\n", "").split(",")
        cur_dict[elements[0]] = np.array(list(map(float, elements[1:])))
    for loss in cur_dict:
        x = np.linspace(0, args.step_x * len(cur_dict[loss]), len(cur_dict[loss]))
        ax.plot(x,  cur_dict[loss], label=loss, color=colors[loss])
    ax.legend()
    ax.grid(b=True, which='major', color='#666666', linestyle='--')
    ax.minorticks_on()
    ax.grid(b=True, which='minor', color='#999999', linestyle='--', alpha=0.2)
    ax.fill_between(x, cur_dict["Dreal.1"], cur_dict["Dreal.9"], color='lightskyblue', alpha=0.5)
    ax.fill_between(x, cur_dict["Dfake.1"], cur_dict["Dfake.9"], color='peachpuff', alpha=0.5)

    plt.savefig(os.path.join(args.path, item, "losses", "logits.png"))

print("Saved losses and logits plots in %s/${exp_name}/losses/" % (args.path))