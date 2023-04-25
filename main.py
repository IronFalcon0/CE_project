from utils import *

from knapsack import Knapsack
from JB_numbers import JB_numbers
import time

def main():
    #knapsack = Knapsack()
    #data = knapsack.run()
    t1 = time.time()
    avg_pen, best_pen = jb_numbers_test('penalize')
    t2 = time.time()
    avg_rep, best_rep = jb_numbers_test('repair')
    t3 = time.time()

    print("Penalize time: ", t2-t1)
    print("Repair time: ", t3-t2)


    title_pen = 'Penalize method'
    title_rep = 'Repair method'
    plot_compare_graphs(avg_pen, best_pen, avg_rep, best_rep, title_pen, title_rep)

    
    

def jb_numbers_test(mode = 'penalize'):
    generations = 50
    pop_size = 1000
    cromo_size = 100
    prob_muta = 0.01
    prob_cross = 0.5
    tour_size = 3
    elite_percent = 0.02
    runs = 1
    

    jb_numbers = JB_numbers(generations, pop_size, cromo_size, prob_muta, prob_cross, runs, tour_size, two_points_cross, elite_percent)
    absolute_data, avg_data = jb_numbers.run(mode)
    data = [[i+1, a] for i, a in enumerate(avg_data)]

    save_data(data, header=["ID", "Fitness"], extra_name='jb_numbers' + '_' + mode)

    return avg_data, absolute_data


if __name__ == '__main__':
    main()