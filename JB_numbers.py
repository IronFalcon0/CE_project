import numpy as np
import random
from operator import itemgetter
from utils import two_points_cross

class JB_numbers():
    def __init__(self, generations, pop_size, cromo_size, prob_muta, prob_cross, runs, tour_size, crossover, elite_percent, fitness):
        self.generations = generations
        self.pop_size = pop_size
        self.cromo_size = cromo_size
        self.prob_muta = prob_muta
        self.prob_cross = prob_cross
        self.tour_size = tour_size
        self.elite_percent = elite_percent
        self.runs = runs

        # default functions/values
        self.tour_seletion = self.tour_sel(tour_size)
        self.crossover = crossover
        self.mutation = self.muta_bin
        self.sel_survivors = self.sel_survivors_elite(elite_percent)
        self.fitness = fitness


    def run(self):
        generations = self.generations
        pop_size = self.pop_size
        cromo_size = self.cromo_size
        prob_muta = self.prob_muta
        prob_cross = self.prob_cross

        runs = self.runs
        fitness_func = self.fitness
        sel_parents = self.tour_seletion
        mutation = self.mutation
        sel_survivors = self.sel_survivors
        recombination = self.crossover



        total_best = []
        best_fit = []
        for _ in range(runs):
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
                #print(progenitores)
                
                # ------ Mutation
                descendentes = []
                for cromo,fit in progenitores:
                    #print(cromo, fit)
                    #continue
                    novo_indiv = mutation(cromo, prob_muta)

                    descendentes.append((novo_indiv,fitness_func(novo_indiv)))

                # New population
                populacao = sel_survivors(populacao,descendentes)
                # Evaluate the new population
                populacao = [(indiv[0], fitness_func(indiv[0])) for indiv in populacao]
                best_indiv.append(self.best_pop(populacao)[1])
                avg_indiv.append(np.mean([indiv[1] for indiv in populacao]))
                print("Generation: ", ng, "Best: ", best_indiv[-1], "Avg: ", avg_indiv[-1])

                

            total_best.append(best_indiv)
            best_fit.append(max(best_indiv))

        absolute_best = [max(idx) for idx in zip(*total_best)]
        avg_best = [float(sum(l))/len(l) for l in zip(*total_best)]

        return absolute_best, avg_best
    
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
    
    # Survivals Selection: elitism
    def sel_survivors_elite(self, elite):
        def elitism(parents,offspring):
            size = len(parents)
            comp_elite = int(size* elite)
            offspring.sort(key=itemgetter(1), reverse=True)
            parents.sort(key=itemgetter(1), reverse=True)
            new_population = parents[:comp_elite] + offspring[:size - comp_elite]
            return new_population
        return elitism