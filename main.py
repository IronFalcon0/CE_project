from utils import *

from knapsack import Knapsack
from JB_numbers import JB_numbers
from sum_of_subsets import SumOfSubsets
import time
import numpy as np
import json


headers = ["Run", "Generation", "Best_fit", "Avg_fit"]


def main():
    #knapsack = Knapsack()
    #data = knapsack.run()
    
    
    #jb_numbers_test()

    #knapsack_test()

    sum_subset_test()

    #redo_plots()


def generate_set(range_start, range_end, number_of_sets, avg_size, filename):

    stddev_set_size = round(avg_size - range_start / 2)

    sets = []
    for i in range(number_of_sets):
        set_size = round(random.normalvariate(avg_size, stddev_set_size))
        if set_size < range_start:
            set_size = 1
        elif set_size > range_end:
            set_size = range_end

        # Generate a set of random integers between 1 and 10 with set_size elements
        sets.append(random.sample(range(range_start, range_end+1), set_size))

    with open(filename, "w") as f:
        
        json.dump(sets, f)


def sum_subset_test(plot = True):
    generations = 2000
    pop_size = 300
    prob_muta = 0.02
    prob_cross = 0.75
    tour_size = 5
    elite_percent = 0.05
    runs = 1

    #range_start = 1
    #range_end = 2000
    #number_of_sets = 500
    #avg_size = 100
    filename = 'sets.txt'
    #generate_set(range_start, range_end, number_of_sets, avg_size, filename)
    
    sumOfSubsets = SumOfSubsets(generations, pop_size, prob_muta, prob_cross, runs, tour_size, two_points_cross, elite_percent, filename)
    
    
    t1 = time.time()

    # mode: penalize
    #mode = 'penalize'
    #best_data_pen, avg_data_pen = sumOfSubsets.run(mode)
    #return
    
    #data = prepare_data(best_data_pen, avg_data_pen)

    #save_data(data, header=headers, extra_name='sum_of_subsets' + '_' + mode)

    #t2 = time.time()

    # mode: repair
    mode = 'repair'
    best_data_rep, avg_data_rep = sumOfSubsets.run(mode)
    data = prepare_data(best_data_rep, avg_data_rep)

    save_data(data, header=headers, extra_name='sum_of_subsets' + '_' + mode)
    return

    avg_pen_avg = np.mean(avg_data_pen, axis=0).tolist()
    best_pen_avg = np.mean(best_data_pen, axis=0).tolist()
    avg_rep_avg = np.mean(avg_data_rep, axis=0).tolist()
    best_rep_avg = np.mean(best_data_rep, axis=0).tolist()

    t3 = time.time()
    
    print("Penalize time: ", t2-t1)
    print("Repair time: ", t3-t2)

    if plot:
        title_pen = 'Penalize method'
        title_rep = 'Repair method'
        plot_compare_graphs(avg_pen_avg, best_pen_avg, avg_rep_avg, best_rep_avg, title_pen, title_rep, 'sum_of_subsets')


def knapsack_test(plot = True):
    generations = 2000
    pop_size = 1000
    number_itens = 500
    prob_muta = 0.02
    prob_cross = 0.75
    tour_size = 5
    elite_percent = 0.05
    runs = 30
    max_value = 50
    
    
    knapsack = Knapsack(generations, pop_size, prob_muta, prob_cross, runs, tour_size, two_points_cross, elite_percent, max_value, number_itens)
    
    t1 = time.time()

    # mode: penalize
    mode = 'penalize'
    best_data_pen, avg_data_pen = knapsack.run(mode)
    
    data = prepare_data(best_data_pen, avg_data_pen)

    save_data(data, header=headers, extra_name='knapsack' + '_' + mode)

    t2 = time.time()

    # mode: repair
    mode = 'repair'
    best_data_rep, avg_data_rep = knapsack.run(mode)
    data = prepare_data(best_data_rep, avg_data_rep)

    save_data(data, header=headers, extra_name='knapsack' + '_' + mode)
    

    avg_pen_avg = np.mean(avg_data_pen, axis=0).tolist()
    best_pen_avg = np.mean(best_data_pen, axis=0).tolist()
    avg_rep_avg = np.mean(avg_data_rep, axis=0).tolist()
    best_rep_avg = np.mean(best_data_rep, axis=0).tolist()

    t3 = time.time()
    
    print("Penalize time: ", t2-t1)
    print("Repair time: ", t3-t2)

    if plot:
        title_pen = 'Penalize method'
        title_rep = 'Repair method'
        plot_compare_graphs(avg_pen_avg, best_pen_avg, avg_rep_avg, best_rep_avg, title_pen, title_rep, 'knapsack')
    
    
    
    

def jb_numbers_test(plot = True):
    generations = 2000
    pop_size = 500
    cromo_size = 300
    prob_muta = 0.01
    prob_cross = 0.7
    tour_size = 5
    elite_percent = 0.05
    runs = 30
    

    jb_numbers = JB_numbers(generations, pop_size, cromo_size, prob_muta, prob_cross, runs, tour_size, two_points_cross, elite_percent)
    
    t1 = time.time()

    # mode: penalize
    #mode = 'penalize'
    #best_data_pen, avg_data_pen = jb_numbers.run(mode)
    #data = prepare_data(best_data_pen, avg_data_pen)

    #save_data(data, header=headers, extra_name='jb_numbers' + '_' + mode)

    best_data_pen, avg_data_pen = load_data('final_jb_numbers_penalize')


    t2 = time.time()

    # mode: repair
    mode = 'repair'
    #best_data_rep, avg_data_rep = jb_numbers.run(mode)
    #data = prepare_data(best_data_rep, avg_data_rep)

    #save_data(data, header=headers, extra_name='jb_numbers' + '_' + mode)
    best_data_rep, avg_data_rep = load_data('final_jb_numbers_repair')


    avg_pen_avg = np.mean(avg_data_pen, axis=0).tolist()
    best_pen_avg = np.mean(best_data_pen, axis=0).tolist()
    avg_rep_avg = np.mean(avg_data_rep, axis=0).tolist()
    best_rep_avg = np.mean(best_data_rep, axis=0).tolist()

    t3 = time.time()
    
    print("Penalize time: ", t2-t1)
    print("Repair time: ", t3-t2)

    if plot:
        title_pen = 'Penalize method'
        title_rep = 'Repair method'
        plot_compare_graphs(avg_pen_avg, best_pen_avg, avg_rep_avg, best_rep_avg, title_pen, title_rep, 'jb_numbers')



def redo_plots():
    # jb_numbers
    best_data_pen, avg_data_pen = load_data('final_jb_numbers_penalize')

    best_data_rep, avg_data_rep = load_data('final_jb_numbers_repair')


    avg_pen_avg = np.mean(avg_data_pen, axis=0).tolist()
    best_pen_avg = np.mean(best_data_pen, axis=0).tolist()
    avg_rep_avg = np.mean(avg_data_rep, axis=0).tolist()
    best_rep_avg = np.mean(best_data_rep, axis=0).tolist()


    title_pen = 'Penalize method'
    title_rep = 'Repair method'
    plot_compare_graphs(avg_pen_avg, best_pen_avg, avg_rep_avg, best_rep_avg, title_pen, title_rep, 'jb_numbers')

    # knapsack
    
    best_data_pen, avg_data_pen = load_data('final_knapsack_penalize')

    best_data_rep, avg_data_rep = load_data('final_knapsack_repair')


    avg_pen_avg = np.mean(avg_data_pen, axis=0).tolist()
    best_pen_avg = np.mean(best_data_pen, axis=0).tolist()
    avg_rep_avg = np.mean(avg_data_rep, axis=0).tolist()
    best_rep_avg = np.mean(best_data_rep, axis=0).tolist()


    title_pen = 'Penalize method'
    title_rep = 'Repair method'
    plot_compare_graphs(avg_pen_avg, best_pen_avg, avg_rep_avg, best_rep_avg, title_pen, title_rep, 'knapsack')


def load_data(filename):
    best_data, avg_data = [], []
    chunks_best, chunks_avg = [], []
    run = -1
    first = True
    with open('results/' + filename, 'r') as f:
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

if __name__ == '__main__':
    main()