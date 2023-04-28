import random
import numpy as np
import math
import copy

from operator import itemgetter


class Knapsack():
    def __init__(self, generations, pop_size, prob_muta, prob_cross, runs, tour_size, crossover, elite_percent, max_value, number_itens):
        self.generations = generations
        self.pop_size = pop_size
        self.prob_muta = prob_muta
        self.prob_cross = prob_cross
        self.tour_size = tour_size
        self.elite = elite_percent
        self.max_value = max_value
        self.runs = runs

        # default functions/values
        self.tour_seletion = self.tour_sel(tour_size)
        self.crossover = crossover

        self.number_itens = number_itens
        self.items = self.generate_uncor(number_itens, max_value)
        print(self.items)
        #exit(0)


    def run(self, mode='penalize'):         
        generations = self.generations
        pop_size = self.pop_size
        prob_muta = self.prob_muta
        prob_cross = self.prob_cross
        max_value = self.max_value

        runs = self.runs
        sel_parents = self.tour_seletion
        mutation = self.muta_bin
        recombination = self.crossover


        if mode == 'penalize':
            fitness_func = self.evaluate_linear
        elif mode == 'repair':
            fitness_func = self.fitness_repair_value_to_profit
        else:
            print("Invalid mode")
            return

        """
        if mode == 'penalize_log':
            fitness_func = self.evaluate_log
        elif mode == 'penalize_linear':
            fitness_func = self.evaluate_linear
        elif mode == 'penalize_quadratic':
            fitness_func = self.evaluate_quadratic
        elif mode == 'repair_value':
            fitness_func = self.fitness_repair_value
        elif mode == 'repair_weight':
            fitness_func = self.fitness_repair_weight
        elif mode == 'repair_value_to_profit':
            fitness_func = self.fitness_repair_value_to_profit
        else:
            print("Invalid mode")
            return
        """
        
        total_best_indiv = []
        total_avg_indiv = []
        for _ in range(runs):
            print("Run: ", _)

            # inicialize population: indiv = (cromo,fit)
            populacao = self.gera_pop(pop_size, self.number_itens)
            
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
                populacao = self.elitism(populacao,descendentes)
                
                # Get the best individual from the generation
                best_indiv.append(self.best_pop(populacao)[1])
                avg_indiv.append(np.mean([indiv[1] for indiv in populacao]))
                #print("Generation: ", ng, "Best: ", best_indiv[-1], "Avg: ", avg_indiv[-1])

                
            total_best_indiv.append(best_indiv)
            total_avg_indiv.append(avg_indiv)

        return total_best_indiv, total_avg_indiv
        

    def fitness(self, indiv):
        # Returns the fitness of an individual
        
        quali = self.evaluate_zero(self.phenotype(indiv))
        return quali
    
    def phenotype(self, indiv):
        # Returns the phenotype of an individual wuth the format [[id, weight, value], ...]
        # indiv has the format [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        pheno = [[id, self.items['weights'][id], self.items['values'][id]] for id in range(len(indiv)) if indiv[id] == 1]
        return pheno
    
    def evaluate_zero(self, pheno):
        # Returns the total value of the items in the knapsack
        total_weight = sum([weight for id,weight,value in pheno])
        capacity = self.items['capacity']
        if total_weight > capacity:
            return 0
        return sum([value for id,weight,value in pheno])
        
    
    # ---------------------------- DATA SET -----------------------------
    
    # este é melhor
    def generate_uncor(self, size_items, max_value):
        # Generates a random data set
        weights = [int(random.uniform(1, max_value)) for i in range(size_items)]
        values = [int(random.uniform(1, max_value)) for i in range(size_items)]
        capacity = int(0.5 * sum(weights))
        return {'weights': weights, 'values': values, 'capacity': capacity}
    
    def generate_weak_cor(self, size_items, max_value, amplitude):
        # Generates a weakly correlated data set
        weights = [random.uniform(1, max_value) for i in range(size_items)]
        values = []
        for i in range(size_items):
            value = weights[i] + random.uniform(-amplitude, amplitude)
            while value <= 0:
                value = weights[i] + random.uniform(-amplitude, amplitude)
            values.append(value)
        capacity = int(0.5 * sum(weights))
        return {'weights': weights, 'values': values, 'capacity': capacity}
    
    def generate_strong_cor(self, size_items, max_value, amplitude):
        # Generates a strongly correlated data set
        weights = [random.uniform(1, max_value) for i in range(size_items)]
        values = [weights[i] + amplitude for i in range(size_items)]
        capacity = int(0.5 * sum(weights))
        return {'weights': weights, 'values': values, 'capacity': capacity}
    
    
    def best_pop(self, populacao):
        populacao.sort(key=itemgetter(1),reverse=True)
        return populacao[0]

    
    
    # ---------------------------- DECODERS -----------------------------
    
    def list_items(self):
        # Returns a list of items with the following format: [id, weight, value, ratio]
        weight_value_list = list(zip(self.items['weights'], self.items['values']))
        l_items = [[i, w, v, v/w] for i, (w, v) in enumerate(weight_value_list)]
        return l_items
    
    def decode_int(self, indiv):
        """
        - Defines the intems in the KP based on a list of integers (indiv)
        - problem is a dictionary with the following keys: weights, values, capacity
        """
        
        # build list of items
        capacity = self.items['capacity']
        l_items = self.list_items()
        sum_weight = 0
        res = []
        for i in range(len(indiv)):
            j = indiv[i]
            # remove from l_items the item of order j
            id, w, v, r = l_items.pop(j)
            if sum_weight + w <= capacity:
                sum_weight += w
                res.append([id, w, v, r])
            else:
                return res
        return res
    
    
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
    

    # Initialize population
    def gera_pop(self, size_pop,size_cromo):
        return [(self.gera_indiv(size_cromo),0) for i in range(size_pop)]
    
    def gera_indiv(self, size_cromo):
        # random initialization
        indiv = [random.randint(0,1) for i in range(size_cromo)]
        return indiv
    
    # ============================ FITNESS ===============================
    
    # ---------------------------- PENALIZE -----------------------------
    
    def evaluate_log(self, pheno):
        # Returns the total value of the items in the knapsack and penalizes the solutions that exceed the capacity
        total_weight = sum([weight for id,weight,value in pheno])
        quality = sum([value for id,weight,value in pheno])
        capacity = self.items['capacity']
        if total_weight > capacity:
            pho = max([v/w for i,w,v in pheno])
            quality -= math.log((total_weight - capacity) * pho + 1,2)
        return quality
    
    def evaluate_linear(self, indiv):
        pheno = self.phenotype(indiv)
        # Returns the total value of the items in the knapsack and penalizes the solutions that exceed the capacity
        total_weight = sum([weight for id,weight,value in pheno])
        quality = sum([value for id,weight,value in pheno])
        capacity = self.items['capacity']
        if total_weight > capacity:
            pho = max([v/w for i,w,v in pheno])
            quality -= (total_weight - capacity) * pho
        return quality
    
    def evaluate_quadratic(self, pheno):
        # Returns the total value of the items in the knapsack and penalizes the solutions that exceed the capacity
        total_weight = sum([weight for id,weight,value in pheno])
        quality = sum([value for id,weight,value in pheno])
        capacity = self.items['capacity']
        if total_weight > capacity:
            pho = max([v/w for i,w,v in pheno])
            quality -= (total_weight - capacity) ** 2 * pho
        return quality
    
    # ---------------------------- REPAIR -----------------------------
    
    def fitness_repair_value(self, indiv):
        indiv = self.repair_value(indiv)
    
        return self.evaluate_zero(self.phenotype(indiv))

    def fitness_repair_weight(self, indiv):
        indiv = self.repair_weight(indiv)
        
        return self.evaluate_zero(self.phenotype(indiv))
    
    def fitness_repair_value_to_profit(self, indiv):
        indiv = self.repair_value_to_profit(indiv)
        
        return self.evaluate_linear(indiv)
    
    
    def repair_value(self, cromo):
        # Returns a valid individual by removing the least valued item from the knapsack
        # indiv = (cromo, fitness)
        indiv = copy.deepcopy(cromo)
        capacity = self.items['capacity']
        pheno = self.phenotype(indiv)
        # sort by value
        pheno.sort(key=itemgetter(2))
        
        weight_indiv = get_weight(indiv, self.items)
        for index, weight, value in pheno:
            if weight_indiv <= capacity:
                break
            indiv[index] = 0
            weight_indiv -= weight
        return indiv
    
    def repair_weight(self, cromo):
        # Returns a valid individual by removing the heaviest item from the knapsack
        indiv = copy.deepcopy(cromo)
        capacity = self.items['capacity']
        pheno = self.phenotype(indiv)
        pheno.sort(key=itemgetter(1))
        
        weight_indiv = get_weight(indiv)
        for index, weight, value in pheno:
            if weight_indiv <= capacity:
                break
            indiv[index] = 0
            weight_indiv -= weight
        return indiv
    
    def repair_value_to_profit(self, cromo):
        # Returns a valid individual by removing the least ratio value/weight item from the knapsack
        indiv = copy.deepcopy(cromo)
        capacity = self.items['capacity']
        pheno = self.phenotype(indiv)
        pheno = [[i, w, v, float(v/w)] for i, w, v in pheno]
        pheno.sort(key= itemgetter(3))
        
        weight_indiv = 100 # get_weight(indiv, problem)
        for index, weight, value, ratio in pheno:
            if weight_indiv <= capacity:
                break
            indiv[index] = 0
            weight_indiv -= weight
        return indiv