import numpy as np
import random
from operator import itemgetter
from utils import two_points_cross
import time

class JB_numbers():
    def __init__(self, generations, pop_size, cromo_size, prob_muta, prob_cross, runs, tour_size, crossover, elite_percent):
        self.generations = generations
        self.pop_size = pop_size
        self.cromo_size = cromo_size
        self.prob_muta = prob_muta
        self.prob_cross = prob_cross
        self.tour_size = tour_size
        self.elite = elite_percent
        self.runs = runs

        # default functions/values
        self.tour_seletion = self.tour_sel(tour_size)
        self.crossover = crossover
        self.mutation = self.muta_bin
        

    # Modes: penalize, repair
    def run(self, mode='penalize'):
    
        generations = self.generations
        pop_size = self.pop_size
        cromo_size = self.cromo_size
        prob_muta = self.prob_muta
        prob_cross = self.prob_cross

        runs = self.runs
        sel_parents = self.tour_seletion
        mutation = self.mutation
        #sel_survivors = self.sel_survivors
        recombination = self.crossover

        if mode == 'penalize':
            fitness_func = self.jb_fitness_penalize
        elif mode == 'repair':
            fitness_func = self.jb_fitness_repair
        else:
            print("Invalid mode")
            return



        total_best_indiv = []
        total_avg_indiv = []
        t1 = time.time()
        for _ in range(runs):
            print("Run: ", _, ", Time: ", time.time()-t1, "s")

            # inicialize population: indiv = (cromo,fit)
            populacao = self.gera_pop(pop_size, cromo_size)
            
            # evaluate population
            populacao = [(indiv[0], fitness_func(indiv[0])) for indiv in populacao]
            best_indiv = []
            avg_indiv = []
            for ng in range(generations):
                # sparents selection
                mate_pool = sel_parents(populacao)
            # Variation
            # ------ Crossover
                progenitores = []
                for i in  range(0,pop_size, 2):
                    indiv_1= mate_pool[i]
                    indiv_2 = mate_pool[i+1]
                    filhos = recombination(indiv_1,indiv_2, prob_cross)
                    progenitores.extend(filhos) 
                
                # ------ Mutation
                descendentes = []
                for cromo,fit in progenitores:
                    novo_indiv = mutation(cromo, prob_muta)

                    descendentes.append((novo_indiv,fitness_func(novo_indiv)))

                # New population
                populacao = self.elitism(populacao, descendentes)

                best_indiv.append(self.best_pop(populacao)[1])

                avg_indiv.append(np.mean([indiv[1] for indiv in populacao]))
                #print("Generation: ", ng, "Best: ", best_indiv[-1], "Avg: ", avg_indiv[-1])

                
            total_best_indiv.append(best_indiv)
            total_avg_indiv.append(avg_indiv)


        return total_best_indiv, total_avg_indiv
    
    # Initialize population
    def gera_pop(self, size_pop,size_cromo):
        return [(self.gera_indiv(size_cromo),0) for i in range(size_pop)]
    
    def gera_indiv(self, size_cromo):
        # random initialization
        indiv = [random.randint(0,1) for i in range(size_cromo)]
        return indiv

    
    def best_pop(self, populacao):
        populacao.sort(key=itemgetter(1),reverse=True)
        return populacao[0]
    

    def tour_sel(self, t_size):
        def tournament(pop):
            size_pop= len(pop)
            mate_pool = []
            for i in range(size_pop):
                winner = self.one_tour(pop,t_size)
                mate_pool.append(winner)
            return mate_pool
        return tournament
    
    def one_tour(self, population,size):
        """Maximization Problem. Deterministic"""
        pool = random.sample(population, size)
        pool.sort(key=itemgetter(1), reverse=True)
        return pool[0]
    
    # Variation operators: Binary mutation	    
    def muta_bin(self, indiv,prob_muta):
        # Mutation by gene

        for i in range(len(indiv)):
            value = random.uniform(0, 1)
            if value < prob_muta:
                indiv[i] = 1 - indiv[i]

        return indiv
    
    # Survivals Selection: elitism
    def elitism(self, parents, offspring):
        size = len(parents)
        comp_elite = int(size* self.elite)

        offspring.sort(key=itemgetter(1), reverse=True)

        
        parents.sort(key=itemgetter(1), reverse=True)
        new_population = parents[:comp_elite] + offspring[:size - comp_elite]
        new_population.sort(key=itemgetter(1), reverse=True)

        return new_population

    


    # --------------------------- JB fitness function ---------------------------
    def jb_fitness_penalize(self, indiv):
        return self.evaluate(self.phenotype(indiv), len(indiv))

    def phenotype(self, indiv):
        fen = [i+1 for i in range(len(indiv)) if indiv[i] == 1]
        return fen


    def evaluate(self, indiv, comp):
        alfa = 1.0
        beta = 1.1
        return alfa * len(indiv) - beta * self.viola(indiv,comp)

    def viola(self, indiv,comp):
        # Count violations
        v = 0
        for elem in indiv:
            limite = min(elem-1,comp-elem)
            vi = 0
            for j in range(1,limite+1):
                if ((elem - j) in indiv) and ((elem+j) in indiv):
                    vi += 1
            v += vi
        return v
    
    # --------------------------- JB repair function ---------------------------

    def jb_fitness_repair(self, indiv):
        indiv = self.jb_repair(indiv)

        return self.evaluate(self.phenotype(indiv), len(indiv))
    
    def count_consecutive_ones(self, indiv):
        numbers_of_ones = []
        count = 0
        for i in range(len(indiv)):
            if indiv[i] == 1:
                count += 1
            else:
                if count > 0:
                    numbers_of_ones.append(count)
                    count = 0
                
        if count > 0:
            numbers_of_ones.append(count)
        return numbers_of_ones

    def find_solution(self, num):
        size = num
        midpoint = size//2
        
        array = [abs(size-i-1) if i > midpoint else abs(i) for i in range(size)]
        indices_remove = [i for i, x in enumerate(array) if (x % 2 == 0 and x > 0) or i == len(array)//2]
        
        return indices_remove
    


    def jb_repair(self, indiv):
        pheno = self.phenotype(indiv)
        while self.viola(pheno, len(indiv)) > 0:
            
            num_ones = self.count_consecutive_ones(indiv)
            count = 0
            id = 0
            while id < len(indiv):
                
                if indiv[id] == 0:
                    id += 1
                    continue
                
                size_group = num_ones[count]
                ids_remove = self.find_solution(size_group)
                for rm_id in ids_remove:
                    indiv[id + rm_id] = 0
                
                id += size_group
                count += 1
            
            pheno = self.phenotype(indiv)
            
        return indiv

