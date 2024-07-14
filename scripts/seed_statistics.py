#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import re
import numpy as np
import matplotlib.pyplot as plt

epoch_pattern = re.compile(r'Epoch (\d+), loss: ([-\d\.e]+)\n')
test_pattern = re.compile(r'Test ability: ([\d\.]+)%, pred time: ([\d\.]+), min time: ([\d\.]+), (\d+) timeouts:.*')

def read_output(path: str):
    with open(path, 'r') as f:
        lines = f.readlines()
    train_losses = np.zeros(100, dtype=np.float32)
    abilities = np.zeros(100, dtype=np.float32)
    pred_times = np.zeros(100, np.float32)
    nums_timeout = np.zeros(100, np.float32)

    for line in lines:
        result = epoch_pattern.match(line)
        if result is not None:
            epoch, train_loss = result.groups()
            epoch = int(epoch)
            train_loss = float(train_loss)
            train_losses[epoch] = train_loss
            continue
        result = test_pattern.match(line)
        if result is not None:
            ability, pred, _, num_timeout = result.groups()
            abilities[epoch] = ability
            pred_times[epoch] = pred
            nums_timeout[epoch] = num_timeout
    print(epoch)
    return train_losses[:epoch + 1], abilities[:epoch + 1], pred_times[:epoch + 1], nums_timeout[:epoch + 1]

def read_outputs(pattern: str, seeds: list[int]):
    ret_train_losses = []
    ret_abilities = []
    ret_pred_times = []
    ret_nums_timeout = []
    for seed in seeds:
        path = pattern.format(seed)
        train_losses, abilities, pred_times, nums_timeout = read_output(path)
        ret_train_losses.append(train_losses)
        ret_abilities.append(abilities)
        ret_pred_times.append(pred_times)
        ret_nums_timeout.append(nums_timeout)
    return np.stack(ret_train_losses), np.stack(ret_abilities), np.stack(ret_pred_times), np.stack(ret_nums_timeout)

def plot_dir(base_dir: str, num_seeds: int):
    train_loss, abilities, _, _ = read_outputs(os.path.join(base_dir, 'output_{}.txt'), range(num_seeds))
    num_seeds, num_epoch = abilities.shape
    abilities_mean = np.mean(abilities, axis=0)
    abilities_std = np.sqrt(np.sum((abilities - abilities_mean) ** 2, axis=0) / (num_seeds - 1))
    train_loss_mean = np.mean(train_loss, axis=0)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=600)
    ax.plot(np.arange(num_epoch) + 1, train_loss_mean, marker='o', linestyle='-', markersize=3, color='black')

    ax.set_title('Mean Training Loss per Epoch')
    ax.set_xlabel('Epoch')
    ax.set_xlim(1, 100)
    ax.set_ylabel('Mean Training Loss')
    ax.grid(True)

    # Tight layout for better spacing
    fig.tight_layout()

    # Save the figure
    fig.savefig(os.path.join(base_dir, 'train_losses.png'))

    fig, ax = plt.subplots(figsize=(8, 6), dpi=600)
    ax.plot(np.arange(num_epoch) + 1, abilities_mean, marker='o', linestyle='-', markersize=3, color='black', label='Mean Ability')
    ax.fill_between(np.arange(num_epoch) + 1, abilities_mean - abilities_std, abilities_mean + abilities_std, color='lightblue', alpha=0.5, label='1 Sigma Range')

    ax.set_title('Mean Ability per Epoch')
    ax.set_xlabel('Epoch')
    ax.set_xlim(1, 100)
    ax.set_ylabel('Mean Ability (%)')
    ax.set_ylim(45, 100)
    ax.grid(True)
    ax.legend()

    # Tight layout for better spacing
    fig.tight_layout()

    # Save the figure
    fig.savefig(os.path.join(base_dir, 'abilities.png'))

    abilities = abilities[:,-10:]
    abilities_mean= np.mean(abilities, axis=1)
    abilities_std = np.sqrt(np.sum((abilities - abilities_mean.reshape(-1, 1)) ** 2, axis=1) / 9)

    fig, ax = plt.subplots(figsize=(num_seeds, 6), dpi=600)
    ax.bar(np.arange(num_seeds), abilities_mean, yerr=abilities_std, capsize=5,
        edgecolor='black', facecolor='none', linewidth=1.5, ecolor='black', error_kw={'elinewidth': 2})

    ax.set_title('Comparison of Mean Values Across Random Seeds')
    ax.set_xlabel('Random Seed')
    ax.set_ylabel('Mean Value')
    ax.set_xticks(np.arange(num_seeds))  # Set x-ticks to show each random seed
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, axis='y')  # Enhance visibility of horizontal grid lines

    # Tight layout for better spacing
    fig.tight_layout()

    # Save the figure
    fig.savefig(os.path.join(base_dir, 'random_seeds_comparison.png'))

def main(args):
    plot_dir(args.dir, args.seeds)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str)
    parser.add_argument('--seeds', type=int, default=10)
    args = parser.parse_args()
    main(args)