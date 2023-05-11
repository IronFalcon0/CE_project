import scipy.stats as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv

import statsmodels.api as sm
from utils import plot_compare_graphs

# ------------------------ UTILS ------------------------ 

def flatten(l):
    return [item for sublist in l for item in sublist]

def dim(data):
    count = 0
    while type(data) == list:
        count += 1
        data = data[0]
    return count


# Parametric??
def test_normal_ks(data):
    """Kolgomorov-Smirnov"""
    if dim(data) > 1:
        data = flatten(data)
    norm_data = (data - np.mean(data))/(np.std(data)/np.sqrt(len(data)))
    return st.kstest(norm_data,'norm')

def test_normal_sw(data):
    """Shapiro-Wilk"""
    norm_data = (data - np.mean(data))/(np.std(data)/np.sqrt(len(data)))
    return st.shapiro(norm_data)

# Non Parametric
def mann_whitney(data1,data2):
    """
    non parametric
    two samples
    independent
    """    
    return st.mannwhitneyu(data1, data2)


def qqplot(data):
    data = np.array(data)
    sm.qqplot(data, line='s')
    plt.show()


def stats(file_pen, file_rep, out_name):
    data_penalize = pd.read_csv(file_pen)
    data_repair = pd.read_csv(file_rep)

    groups_penalize_best = []
    groups_repair_best = []

    for g_n, group_data in data_penalize.groupby('Run'):
        groups_penalize_best.append(group_data.iloc[len(group_data)-1, 2])
    
    for g_n, group_data in data_repair.groupby('Run'):
        groups_repair_best.append(group_data.iloc[len(group_data)-1, 2])


    # normality tests
    print("Normality Shapiro-Wilk test, method penalize: ", test_normal_ks(groups_penalize_best))
    print("Normality Shapiro-Wilk test, method repair: ", test_normal_ks(groups_repair_best))

    print("Repair fitness:", groups_repair_best)
    print("Penalize fitness:", groups_penalize_best)

    # plot histograms distributions
    plt.hist(groups_penalize_best, bins=20)
    plt.savefig(f'plots/{out_name}_penalize.png')
    plt.clf()
    
    plt.hist(groups_repair_best, bins=20)
    plt.savefig(f'plots/{out_name}_repair.png')
    plt.clf()
    

    # non parametric tests
    print("Mann-Whitney U test: ")
    print(st.mannwhitneyu(groups_repair_best, groups_penalize_best, alternative='greater'))
    print()
    
    #qqplot(groups_penalize_best)
    #qqplot(groups_repair_best)


def load_data(filename):
    best_data, avg_data = [], []
    chunks_best, chunks_avg = [], []
    run = -1
    first = True
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if first:
                first = False
                continue
            if row[0] != run:
                if len(chunks_best) > 1:
                    best_data.append(chunks_best)
                    avg_data.append(chunks_avg)
                chunks_best, chunks_avg = [], []
                run = row[0]
            chunks_best.append(float(row[2]))
            chunks_avg.append(float(row[3]))

    if len(chunks_best) > 1:
        best_data.append(chunks_best)
        avg_data.append(chunks_avg)

    return best_data, avg_data


def redo_plots(file_pen, file_rep, out_name, axis_plot=None):
    title_pen = 'Penalize method'
    title_rep = 'Repair method'

    best_data_pen, avg_data_pen = load_data(file_pen)
    best_data_rep, avg_data_rep = load_data(file_rep)

    avg_pen_avg = np.mean(avg_data_pen, axis=0).tolist()
    best_pen_avg = np.mean(best_data_pen, axis=0).tolist()
    avg_rep_avg = np.mean(avg_data_rep, axis=0).tolist()
    best_rep_avg = np.mean(best_data_rep, axis=0).tolist()

    plot_compare_graphs(avg_pen_avg, best_pen_avg, avg_rep_avg, best_rep_avg, title_pen, title_rep, out_name, axis_plot)


def main():

    if input("Create fitness plots (y/n): ") == 'y':
        axis_plot = {'x1_min': None, 'x1_max': None, 'x2_min': None, 'x2_max': None, 'y1_min': -400, 'y1_max': 50, 'y2_min': None, 'y2_max': None}
        redo_plots('results/final_jb_numbers_penalize', 'results/final_jb_numbers_repair', 'jb_numbers', axis_plot)

        axis_plot = {'x1_min': None, 'x1_max': None, 'x2_min': None, 'x2_max': None, 'y1_min': None, 'y1_max': 10000, 'y2_min': None, 'y2_max': None}
        redo_plots('results/final_knapsack_penalize', 'results/final_knapsack_repair', 'knapsack', axis_plot)

        axis_plot = {'x1_min': None, 'x1_max': None, 'x2_min': None, 'x2_max': None, 'y1_min': 800, 'y1_max': 925, 'y2_min': 800, 'y2_max': 925}
        redo_plots('results/final_min_set_cover_penalize', 'results/final_min_set_cover_repair', 'min_set_cover', axis_plot)

        print("Plots created.")

    if input("Test jb_numbers (y/n): ") == 'y':
        stats('results/final_jb_numbers_penalize', 'results/final_jb_numbers_repair', 'jb_numbers')

    if input("Test knapsack (y/n): ") == 'y':
        stats('results/final_knapsack_penalize', 'results/final_knapsack_repair', 'knapsack')

    if input("Test min_set_cover (y/n): ") == 'y':
        stats('results/final_min_set_cover_penalize', 'results/final_min_set_cover_repair', 'min_set_cover')






if __name__ == '__main__':
    main()