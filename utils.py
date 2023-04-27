import os
from datetime import datetime
import matplotlib.pyplot as plt
import csv
import random

# -------------------------------- UTILS --------------------------------
def frange(n1,n2=None,n3=1.0):
    """
    Range with floats.
    Can be called as we do with range:
    frange(n), frange(n1,n2), fange(n1,n2,n3)
    """
    if n2 == None:
        n2 = n1
        n1 = 0.0
    nextn = n1
    while (n3 >= 0.0 and nextn <= n2) or (n3 < 0.0 and nextn >= n2):
        yield nextn
        nextn += n3


def derivative(func, delta=0.0001):
    def der(x):
        return (func(x+delta) - func(x))/delta
    return der


def best_pop(self, populacao):
    populacao.sort(key=itemgetter(1),reverse=True)
    return populacao[0]



# ---------------------------- VARIATION OPERATORS -----------------------
def two_points_cross(indiv_1, indiv_2,prob_cross):
    value = random.random()
    if value < prob_cross:
        cromo_1 = indiv_1[0]
        cromo_2 = indiv_2[0]	    
        pc= random.sample(range(len(cromo_1)),2)
        pc.sort()
        pc1,pc2 = pc
        f1= cromo_1[:pc1] + cromo_2[pc1:pc2] + cromo_1[pc2:]
        f2= cromo_2[:pc1] + cromo_1[pc1:pc2] + cromo_2[pc2:]
        return ((f1,0),(f2,0))
    else:
        return (indiv_1,indiv_2)
    
def uniform_cross(indiv_1, indiv_2, prob_cross):
    value = random.random()
    
    if value < prob_cross:
        cromo_1 = indiv_1[0]
        cromo_2 = indiv_2[0]
        f1 = []
        f2 = []
        for i in range(0, len(cromo_1)):
            if random() < 0.5:
                f1.append(cromo_1[i])
                f2.append(cromo_2[i])
            else:
                f1.append(cromo_2[i])
                f2.append(cromo_1[i])
        return ((f1,0),(f2,0))
    else:
        return (indiv1,indiv2)
    
    
       
# Variation operators: Binary mutation	    
def muta_bin(self, indiv,prob_muta):
    # Mutation by gene
    cromo = indiv[:]
    for i in range(len(indiv)):
        cromo[i] = self.muta_bin_gene(cromo[i],prob_muta)

    return indiv

def muta_bin_gene(self, gene, prob_muta):
    g = gene
    value = random.random()
    if value < prob_muta:
        g ^= 1
    return g

# ---------------------------- PARENTS SELECTION -----------------------------

def sel_survivors_elite(self, elite):
    def elitism(parents,offspring):
        size = len(parents)
        comp_elite = int(size* elite)
        offspring.sort(key=itemgetter(1), reverse=True)
        parents.sort(key=itemgetter(1), reverse=True)
        new_population = parents[:comp_elite] + offspring[:size - comp_elite]
        return new_population
    return elitism

# ---------------------------- VISUALIZATION -----------------------------
def display_function(f, x_min, x_max, delta=0.1):
    x = list(frange(x_min, x_max,delta))
    y = [f(i) for i in x]
    plt.title(f.__name__)
    plt.grid(True)
    plt.axhline(c='black')
    plt.axvline(c='black')    
    plt.xlabel('X')
    plt.ylabel('Y= '+f.__name__ + '(X)')
    plt.plot(x,y, 'r')
    plt.show()
    
def display_all_function(f, x_min, x_max, delta=0.1):
    x = list(frange(x_min, x_max,delta))
    y_1 = [f(i) for i in x]
    y_2 = [derivative(f)(i) for i in x]
    y_3 = [derivative(derivative(f))(i) for i in x]
    plt.title(f.__name__)
    plt.grid(True)
    plt.axhline(c='black')
    plt.axvline(c='black')    
    plt.xlabel('X')
    plt.ylabel('Y= '+f.__name__ + '(X)')
    plt.plot(x,y_1, label='Function')
    plt.plot(x,y_2, label='First Deriv.')
    plt.plot(x,y_3, label=' Second Deriv.')
    plt.legend(loc='best')
    plt.show()
    
def display_data(data):
    x = list(range(len(data)))
    plt.grid(True)
    plt.axhline(c='black')
    plt.axvline(c='black')    
    plt.plot(x,data, 'r')
    plt.show()  
    

def plot_compare_graphs(avg_fitness1, best1, avg_fitness2, best2, title1, title2):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    ax1.grid(True)
    ax1.axhline(c='black')
    ax1.axvline(c='black')
    ax1.plot(avg_fitness1, 'b', label='Avg Fitness')
    ax1.plot(best1, 'r', label='Best')
    ax1.set_ylim([-50, max(best1) + 10])
    ax1.set_xlim([0, len(best1)])
    ax1.set_title(title1)
    ax1.legend()

    ax2.grid(True)
    ax2.axhline(c='black')
    ax2.axvline(c='black')
    ax2.plot(avg_fitness2, 'b', label='Avg Fitness')
    ax2.plot(best2, 'r', label='Best')
    ax2.set_ylim([-50, max(best2) + 10])
    ax2.set_xlim([0, len(best2)])
    ax2.set_title(title2)
    ax2.legend()
    
    plt.show()


# ---------------------------- SAVE DATA -----------------------------
def save_data(data, header=[], extra_name='', path='results'):
    
    if extra_name:
        folder_name = datetime.now().strftime("%d_%m_%Y_%H_%M") + "_" + extra_name
    else:
        folder_name = datetime.now().strftime("%d_%m_%Y_%H_%M")

    if os.path.exists(os.path.join(path, folder_name)):
        print("File exists: {}\nData not saved".format(folder_name))
        return
    
    with open(os.path.join(path, folder_name), 'w', newline='') as f:
        writer = csv.writer(f)
        if len(header) > 0:
            writer.writerow(header)
        
        writer.writerows(data)