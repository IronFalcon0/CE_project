from utils import *

from knapsack import Knapsack
from JB_numbers import JB_numbers

def main():
    #knapsack = Knapsack()
    #data = knapsack.run()

    jb_numbers_test()

    
    

def jb_numbers_test():
    generations = 30
    pop_size = 100
    cromo_size = 100
    prob_muta = 0.01
    prob_cross = 0.5
    tour_size = 3
    elite_percent = 0.02
    runs = 1

    jb_numbers = JB_numbers(generations, pop_size, cromo_size, prob_muta, prob_cross, runs, tour_size, two_points_cross, elite_percent, fitness)
    absolute_data, avg_data = jb_numbers.run()
    data = [[i+1, a] for i, a in enumerate(avg_data)]

    save_data(data, header=["ID", "Fitness"], extra_name='jb_numbers')


if __name__ == '__main__':
    main()