
import os, sys
sys.path.insert(0, "../stable_baseline3")
from stable_baselines3.common import results_plotter
import gym
import numpy as np
import argparse
import os
import pickle
from copy import deepcopy

import pytablewriter
import seaborn
from matplotlib import pyplot as plt
from scipy.spatial import distance_matrix
from scipy.signal import savgol_filter

from stable_baselines3 import TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import LoadMonitorResultsError, load_results
from stable_baselines3.common.results_plotter import X_EPISODES, X_TIMESTEPS, X_WALLTIME, ts2xy, window_func
parser = argparse.ArgumentParser("Gather results, plot training reward/success")
parser.add_argument("-a", "--algo", help="Algorithm to include", type=str, required=True)
parser.add_argument("-e", "--env", help="Environment to include", type=str, required=True)
parser.add_argument("-f", "--exp-folder", help="Folders to include", type=str, default="./logs")
parser.add_argument("-ids", "--ids", help="indexes of the experiment", nargs="+", type=int)
parser.add_argument("--figsize", help="Figure size, width, height in inches.", nargs=2, type=int, default=[6.4, 4.8])
parser.add_argument("--fontsize", help="Font size", type=int, default=6)
parser.add_argument("-max", "--max-timesteps", help="Max number of timesteps to display", type=int)
parser.add_argument("-x", "--x-axis", help="X-axis", choices=["steps", "episodes", "time"], type=str, default="steps")
parser.add_argument("-y", "--y-axis", help="Y-axis", choices=["success", "reward"], type=str, default="reward")
parser.add_argument("-w", "--episode-window", help="Rolling window size", type=int, default=100)
parser.add_argument("-l", "--labels", help="Label for each folder", nargs="+", type=str)
parser.add_argument("--no-million", action="store_true", default=False, help="Do not convert x-axis to million")
parser.add_argument("--smooth", help="smooth y axis with window size args.smooth", type=int, default=-1)
parser.add_argument("--avg", action='store_true')
args = parser.parse_args()

if not os.path.isdir("./evals"):
    os.mkdir("./evals")

NPY_DIR = "./npy"


# Activate seaborn
seaborn.set()
results = {}
post_processed_results = {}

algos = [args.algo]
exp_folders = [args.exp_folder]
if args.labels is None:
    args.labels = exp_folders
envs = [args.env]

for env in envs:
    plt.figure(f"Results {env}")
    plt.title(f"{env} eval", fontsize=14)

    x_label_suffix = "" if args.no_million else "(in Million)"
    plt.xlabel(f"Timesteps {x_label_suffix}", fontsize=14)
    plt.ylabel("Score", fontsize=14)
    results[env] = {}
    post_processed_results[env] = {}
    for algo in algos:
        for folder_idx, exp_folder in enumerate(exp_folders
    ):

            log_path = os.path.join(exp_folder, algo.lower())

            if not os.path.isdir(log_path):
                continue
            if args.ids is not None:
                dirs = [
                    os.path.join(log_path, d)
                    for d in os.listdir(log_path)
                    if (env in d and os.path.isdir(os.path.join(log_path, d)) and int(d.split("_")[-1]) in args.ids)
                ]
            else:
                dirs = [
                    os.path.join(log_path, d)
                    for d in os.listdir(log_path)
                    if (env in d and os.path.isdir(os.path.join(log_path, d)))
                ]

            logs = []
            for _, dir_ in enumerate(dirs):

                try:
                    log = np.load(os.path.join(dir_, "evaluations.npz"))
                    logs.append(log)
                except FileNotFoundError:
                    print("Eval not found for", dir_)
                    continue
                '''
                for key, item in log.items():
                    print(key)
                    print(item.shape)
                '''
                # log["results"] is evaled multiple times at given timestep, so take mean along axis=1
                if not args.avg:
                    if args.smooth != -1:
                        y = savgol_filter(log['results'].mean(axis=1), args.smooth, 3) # window size args.smooth polynomial order 3

                        plt.plot(log["timesteps"][0:args.max_timesteps], y[:args.max_timesteps], label=dir_.split("/")[-1])
                    else:
                        plt.plot(log["timesteps"][0:args.max_timesteps], log["results"].mean(axis=1)[:args.max_timesteps], label=dir_.split("/")[-1])
                    fname = dir_.split('/')[-1]
                    save_dir = f'./npy/{algo}'
                                        
                    if not os.path.isdir(save_dir):
                        os.mkdir(save_dir)
                    save_f = os.path.join(save_dir, f"{fname}.npy")
                    np.save(save_f, log["results"].mean(axis=1)[:args.max_timesteps])
            print(logs[0]['timesteps'].shape)
            if args.avg:
                timesteps = np.mean([_log['timesteps'] for _log in logs], axis=0)
                results = np.mean([_log['results'].mean(axis=1) for _log in logs], axis=0)
                plt.plot(timesteps[0:args.max_timesteps], results[:args.max_timesteps])
            plt.legend()
            if not os.path.isdir(f"./evals/{env}"):
                os.makedirs(f"./evals/{env}")
            
            plt.savefig(f"./evals/{env}/{algo}_eval.png")           





algo = args.algo
env = args.env
log_path = os.path.join(args.exp_folder, algo)

x_axis = {"steps": X_TIMESTEPS, "episodes": X_EPISODES, "time": X_WALLTIME}[args.x_axis]
x_label = {"steps": "Timesteps", "episodes": "Episodes", "time": "Walltime (in hours)"}[args.x_axis]

y_axis = {"success": "is_success", "reward": "r"}[args.y_axis]
y_label = {"success": "Training Success Rate", "reward": "Training Episodic Reward"}[args.y_axis]
if args.ids is not None:
    dirs = [
        os.path.join(log_path, folder)
        for folder in os.listdir(log_path)
        if (env in folder and os.path.isdir(os.path.join(log_path, folder)) and int(folder.split("_")[-1]) in args.ids)
    ]
else:
    dirs = [
        os.path.join(log_path, folder)
        for folder in os.listdir(log_path)
        if (env in folder and os.path.isdir(os.path.join(log_path, folder)))
    ]


plt.figure(y_label, figsize=args.figsize)
plt.title(y_label, fontsize=args.fontsize)
plt.xlabel(f"{x_label}", fontsize=args.fontsize)
plt.ylabel(y_label, fontsize=args.fontsize)
for folder in dirs:
    try:
        data_frame = load_results(folder)
    except LoadMonitorResultsError:
        continue
    if args.max_timesteps is not None:
        data_frame = data_frame[data_frame.l.cumsum() <= args.max_timesteps]
        
    try:
        y = np.array(data_frame[y_axis])
    except KeyError:
        print(f"No data available for {folder}")
        continue
    x, _ = ts2xy(data_frame, x_axis)
    print(x)
    # Do not plot the smoothed curve at all if the timeseries is shorter than window size.
    if x.shape[0] >= args.episode_window:
        # Compute and plot rolling mean with window of size args.episode_window
        x, y_mean = window_func(x, y, args.episode_window, np.mean)
        # print(x)
        plt.plot(x, y_mean, linewidth=1.5, label=folder.split("/")[-1])

# plt.legend()
plt.tight_layout()

plt.savefig(f"./evals/{env}/{algo}_train.png")