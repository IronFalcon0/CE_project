import scipy.stats as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import statsmodels.api as sm

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

def main():

    if input("Test jb_numbers (y/n): ") == 'y':
        data_penalize = pd.read_csv('results/final_jb_numbers_penalize')
        data_repair = pd.read_csv('results/final_jb_numbers_repair')

        groups_penalize_avg = []
        groups_penalize_best = []
        groups_repair_avg = []
        groups_repair_best = []

        for g_n, group_data in data_penalize.groupby('Run'):
            groups_penalize_best.append(group_data.iloc[len(group_data)-1, 2])
            groups_penalize_avg.append(group_data.iloc[len(group_data)-1, 3])
        
        for g_n, group_data in data_repair.groupby('Run'):
            groups_repair_best.append(group_data.iloc[len(group_data)-1, 2])
            groups_repair_avg.append(group_data.iloc[len(group_data)-1, 3])



        # normality tests
        print("groups_penalize_best: ", test_normal_ks(groups_penalize_best))
        print("groups_penalize_avg: ", test_normal_ks(groups_penalize_avg))
        print("groups_repair_best: ", test_normal_ks(groups_repair_best))
        print("groups_repair_avg: ", test_normal_ks(groups_repair_avg))

        #print(groups_repair_best)
        #print(groups_penalize_best)

        # plot histograms distributions
        plt.hist(groups_penalize_best, bins=20)
        plt.savefig('plots/jb_numbers_penalize_best.png')
        plt.clf()
        
        plt.hist(groups_penalize_avg, bins=20)
        plt.savefig('plots/jb_numbers_penalize_avg.png')
        plt.clf()
        
        plt.hist(groups_repair_best, bins=20)
        plt.savefig('plots/jb_numbers_repair_best.png')
        plt.clf()
        
        plt.hist(groups_repair_avg, bins=20)
        plt.savefig('plots/jb_numbers_repair_avg.png')
        plt.clf()
        


        # non parametric tests
        print(mann_whitney(groups_penalize_best, groups_repair_best))
        print(mann_whitney(groups_penalize_avg, groups_repair_avg))

        qqplot(groups_penalize_best)
        qqplot(groups_repair_best)

    if input("Test knapsack (y/n): ") == 'y':
        data_penalize = pd.read_csv('results/final_knapsack_penalize')
        data_repair = pd.read_csv('results/final_knapsack_repair')

        groups_penalize_avg = []
        groups_penalize_best = []
        groups_repair_avg = []
        groups_repair_best = []

        for g_n, group_data in data_penalize.groupby('Run'):
            groups_penalize_best.append(group_data.iloc[len(group_data)-1, 2])
            groups_penalize_avg.append(group_data.iloc[len(group_data)-1, 3])
        
        for g_n, group_data in data_repair.groupby('Run'):
            groups_repair_best.append(group_data.iloc[len(group_data)-1, 2])
            groups_repair_avg.append(group_data.iloc[len(group_data)-1, 3])



        # normality tests
        print("groups_penalize_best: ", test_normal_ks(groups_penalize_best))
        print("groups_penalize_avg: ", test_normal_ks(groups_penalize_avg))
        print("groups_repair_best: ", test_normal_ks(groups_repair_best))
        print("groups_repair_avg: ", test_normal_ks(groups_repair_avg))

        #print(groups_repair_best)
        #print(groups_penalize_best)

        # plot histograms distributions
        plt.hist(groups_penalize_best, bins=20)
        plt.savefig('plots/knapsack_penalize_best.png')
        plt.clf()
        
        plt.hist(groups_penalize_avg, bins=20)
        plt.savefig('plots/knapsack_penalize_avg.png')
        plt.clf()
        
        plt.hist(groups_repair_best, bins=20)
        plt.savefig('plots/knapsack_repair_best.png')
        plt.clf()
        
        plt.hist(groups_repair_avg, bins=20)
        plt.savefig('plots/knapsack_repair_avg.png')
        plt.clf()
        


        # non parametric tests
        print(mann_whitney(groups_penalize_best, groups_repair_best))
        print(mann_whitney(groups_penalize_avg, groups_repair_avg))






if __name__ == '__main__':
    main()