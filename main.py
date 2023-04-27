from utils import *

from knapsack import Knapsack
from JB_numbers import JB_numbers
import time
import numpy as np


headers = ["Run", "Generation", "Best_fit", "Avg_fit"]


def main():
    #knapsack = Knapsack()
    #data = knapsack.run()
    
    
    #jb_numbers_test()

    knapsack_test()


def knapsack_test(plot = True):
    generations = 200
    pop_size = 100
    number_itens = 50
    prob_muta = 0.01
    prob_cross = 0.7
    tour_size = 5
    elite_percent = 0.02
    runs = 3
    max_value = 100
    
    
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
        plot_compare_graphs(avg_pen_avg, best_pen_avg, avg_rep_avg, best_rep_avg, title_pen, title_rep)
    
    
    
    

def jb_numbers_test(plot = True):
    generations = 300
    pop_size = 500
    cromo_size = 100
    prob_muta = 0.01
    prob_cross = 0.7
    tour_size = 5
    elite_percent = 0.05
    runs = 30
    

    jb_numbers = JB_numbers(generations, pop_size, cromo_size, prob_muta, prob_cross, runs, tour_size, two_points_cross, elite_percent)
    
    t1 = time.time()

    # mode: penalize
    mode = 'penalize'
    best_data_pen, avg_data_pen = jb_numbers.run(mode)
    data = prepare_data(best_data_pen, avg_data_pen)

    save_data(data, header=headers, extra_name='jb_numbers' + '_' + mode)

    t2 = time.time()

    # mode: repair
    mode = 'repair'
    best_data_rep, avg_data_rep = jb_numbers.run(mode)
    data = prepare_data(best_data_rep, avg_data_rep)

    save_data(data, header=headers, extra_name='jb_numbers' + '_' + mode)
    

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
        plot_compare_graphs(avg_pen_avg, best_pen_avg, avg_rep_avg, best_rep_avg, title_pen, title_rep)





if __name__ == '__main__':
    main()