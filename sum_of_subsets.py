import random
import numpy as np
import math
import copy
import json

from operator import itemgetter
import time
import itertools


class SumOfSubsets():
    def __init__(self, generations, pop_size, prob_muta, prob_cross, runs, tour_size, crossover, elite_percent, sets_file):
        self.generations = generations
        self.pop_size = pop_size
        self.prob_muta = prob_muta
        self.prob_cross = prob_cross
        self.tour_size = tour_size
        self.elite = elite_percent
        self.runs = runs

        # default functions/values
        self.tour_seletion = self.tour_sel(tour_size)
        self.crossover = crossover


        self.sets = self.load_sets(sets_file)


    def run(self, mode='penalize'):         
        generations = self.generations
        pop_size = self.pop_size
        prob_muta = self.prob_muta
        prob_cross = self.prob_cross

        runs = self.runs
        sel_parents = self.tour_seletion
        mutation = self.muta_bin
        recombination = self.crossover


        if mode == 'penalize':
            fitness_func = self.fitness_penalize
        elif mode == 'repair':
            fitness_func = self.fitness_repair
        else:
            print("Invalid mode")
            return


        
        total_best_indiv = []
        total_avg_indiv = []
        t1 = time.time()
        for _ in range(runs):
            print("Run: ", _, ", Time: ", time.time()-t1, "s")

            # inicialize population: indiv = (cromo,fit)
            populacao = self.gera_pop(pop_size)

            
            # evaluate population
            populacao = [(indiv[0], fitness_func(indiv[0])) for indiv in populacao]
            #return
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
                populacao = self.elitism(populacao,descendentes)
                
                # Get the best individual from the generation
                best_indiv.append(self.best_pop(populacao)[1])
                avg_indiv.append(np.mean([indiv[1] for indiv in populacao]))
                
                best_pheno = self.pheno(self.best_pop(populacao)[0])
                if ng % 100 == 0:

                    print("Generation: ", ng, "Best: ", best_indiv[-1], "Avg: ", avg_indiv[-1])

                
            total_best_indiv.append(best_indiv)
            total_avg_indiv.append(avg_indiv)

        return total_best_indiv, total_avg_indiv
    

    def best_pop(self, populacao):
        populacao.sort(key=itemgetter(1),reverse=True)
        return populacao[0]
        

    def gera_pop(self, pop_size):
        populacao = []
        for i in range(pop_size):
            cromo = self.gera_indiv()
            populacao.append((cromo, 0))
        return populacao
    
    def gera_indiv(self):
        return [random.randint(0,1) for i in range(len(self.sets))]


    def load_sets(self, sets_file):
        with open(sets_file, 'r') as f:
            sets = json.loads(f.read())
        return sets

    def pheno(self, cromo):
        indiv_sets = [self.sets[i] for i in range(len(cromo)) if cromo[i] == 1]
        pheno = list(set(itertools.chain(*indiv_sets)))

        return pheno, len(indiv_sets)

    # ----------------- Fitness functions -----------------

    def fitness_penalize(self, indiv):
        pheno, size = self.pheno(indiv)
        fitness = len(pheno) - (size * 2)
        return fitness
    

    def fitness_repair(self, indiv):
        # personlized calculation of fenotype, search for repeated sets and remove them
        indiv_sets = [self.sets[i] for i in range(len(indiv)) if indiv[i] == 1]
        
        ids_rmv = []
        for i in range(len(indiv_sets)):
            for j in range(i+1, len(indiv_sets)):
                set1 = set(indiv_sets[i])
                set2 = set(indiv_sets[j])

                if set1.issubset(set2):
                    ids_rmv.append(i)
                elif set2.issubset(set1):
                    ids_rmv.append(j)


        ids_rmv.sort(reverse=True)


        indiv_sets = [indiv_sets[i] for i in range(len(indiv_sets)) if i not in ids_rmv]

        
        pheno = list(set(itertools.chain(*indiv_sets)))
        size = len(indiv_sets)
        fitness = len(pheno) - (size * 2)
        return fitness




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