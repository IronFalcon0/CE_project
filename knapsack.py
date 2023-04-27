
class Knapsack():
    def __init__(self, generations, pop_size, cromo_size, prob_muta, prob_cross, runs, tour_size, crossover, mutation, sel_survivors, elite_percent, fitness, max_value, max_weight):
        self.generations = generations
        self.pop_size = pop_size
        self.cromo_size = cromo_size
        self.prob_muta = prob_muta
        self.prob_cross = prob_cross
        self.tour_size = tour_size
        self.elite_percent = elite_percent
        self.max_value = max_value
        self.max_weight = max_weight
        self.runs = runs

        # default functions/values
        self.tour_seletion = self.tour_sel(tour_size)
        self.crossover = crossover
        self.mutation = self.mutation
        self.sel_survivors = self.sel_survivors
        self.fitness = fitness

    def run(self, mode='penalize'):         
        generations = self.generations
        pop_size = self.pop_size
        cromo_size = self.cromo_size
        prob_muta = self.prob_muta
        prob_cross = self.prob_cross
        max_weight = self.max_weight
        max_value = self.max_value

        runs = self.runs
        sel_parents = self.tour_seletion
        mutation = self.mutation
        sel_survivors = self.sel_survivors
        recombination = self.crossover

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
        
        total_best = []
        best_fit = []
        for _ in range(runs):
            # inicialize population: indiv = (cromo,fit)
            populacao = self.generate_uncor(pop_size, max_value)
            
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
                # Get the best individual from the generation
                best_indiv.append(self.best_pop(populacao)[1])
                avg_indiv.append(np.mean([indiv[1] for indiv in populacao]))
                print("Generation: ", ng, "Best: ", best_indiv[-1], "Avg: ", avg_indiv[-1])

                
            # Get the best individual from the run
            total_best.append(best_indiv)
            best_fit.append(max(best_indiv))

        absolute_best = [max(idx) for idx in zip(*total_best)]
        avg_best = [float(sum(l))/len(l) for l in zip(*total_best)]

        return best_indiv, avg_indiv
        

    def merito(problem):
        def fitness(indiv):
            # Returns the fitness of an individual
            
            quali = evaluate_zero(phenotype(indiv))
            return quali
        return fitness
    
    def phenotype(indiv):
        # Returns the phenotype of an individual wuth the format [[id, weight, value], ...]
        # indiv has the format [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        pheno = [[id, problem['weights'][id], problem['values'][id]] for id in range(len(indiv)) if indiv[id] == 1]
        return pheno
    
    def evaluate_zero(feno):
        # Returns the total value of the items in the knapsack
        total_weight = sum([weight for id,weight,value in feno])
        capacity = problem['capacity']
        if total_weight > capacity:
            return 0
        return sum([value for id,weight,value in feno])
        
    
    # ---------------------------- DATA SET -----------------------------
    
    # este é melhor
    def generate_uncor(size_items, max_value):
        # Generates a random data set
        weights = [random.uniform(1, max_value) for i in range(size_items)]
        values = [random.uniform(1, max_value) for i in range(size_items)]
        capacity = int(0.5 * sum(weights))
        return {'weights': weights, 'values': values, 'capacity': capacity}
    
    def generate_weak_cor(size_items, max_value, amplitude):
        # Generates a weakly correlated data set
        weight = [random.uniform(1, max_value) for i in range(size_items)]
        values = []
        for i in range(size_items):
            value = weights[i] + random.uniform(-amplitude, amplitude)
            while value <= 0:
                value = weights[i] + random.uniform(-amplitude, amplitude)
            values.append(value)
        capacity = int(0.5 * sum(weights))
        return {'weights': weights, 'values': values, 'capacity': capacity}
    
    def generate_strong_cor(size_items, max_value, amplitude):
        # Generates a strongly correlated data set
        weight = [random.uniform(1, max_value) for i in range(size_items)]
        values = [weights[i] + amplitude for i in range(size_items)]
        capacity = int(0.5 * sum(weights))
        return {'weights': weights, 'values': values, 'capacity': capacity}
    
    
    def best_pop(self, populacao):
        populacao.sort(key=itemgetter(1),reverse=True)
        return populacao[0]

    
    
    # ---------------------------- DECODERS -----------------------------
    
    def list_items(problem):
        # Returns a list of items with the following format: [id, weight, value, ratio]
        weight_value_list = list(zip(problem['weights'], problem['values']))
        l_items = [[i, w, v, v/w] for i, (w, v) in enumerate(weight_value_list)]
        return l_items
    
    def decode_int(indiv, problem):
        """
        - Defines the intems in the KP based on a list of integers (indiv)
        - problem is a dictionary with the following keys: weights, values, capacity
        """
        
        # build list of items
        capacity = problem['capacity']
        l_items = list_items(problem)
        sum_weight = 0
        res = []
        for i in range(len(indiv)):
            j = indiv[i]
            # remove from l_items the item of order j
            id, w, v, r = l_items.pop(j)
            if sum_weight + w <= capacity:
                sum_weights += w
                res.append([id, w, v, r])
            else:
                return res
        return res
    
    # ============================ FITNESS ===============================
    
    # ---------------------------- PENALIZE -----------------------------
    
    def evaluate_log(feno):
        # Returns the total value of the items in the knapsack and penalizes the solutions that exceed the capacity
        total_weight = sum([weight for id,weight,value in feno])
        quality = sum([value for id,weight,value in feno])
        capacity = problem['capacity']
        if total_weight > capacity:
            pho = max([v/w for i,w,v in feno])
            quality -= math.log((total_weight - capacity) * pho + 1,2)
        return quality
    
    def evaluate_linear(feno):
        # Returns the total value of the items in the knapsack and penalizes the solutions that exceed the capacity
        total_weight = sum([weight for id,weight,value in feno])
        quality = sum([value for id,weight,value in feno])
        capacity = problem['capacity']
        if total_weight > capacity:
            pho = max([v/w for i,w,v in feno])
            quality -= (total_weight - capacity) * pho
        return quality
    
    def evaluate_quadratic(feno):
        # Returns the total value of the items in the knapsack and penalizes the solutions that exceed the capacity
        total_weight = sum([weight for id,weight,value in feno])
        quality = sum([value for id,weight,value in feno])
        capacity = problem['capacity']
        if total_weight > capacity:
            pho = max([v/w for i,w,v in feno])
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
        
        return self.evaluate_zero(self.phenotype(indiv))
    
    
    def repair_value(cromo):
        # Returns a valid individual by removing the least valued item from the knapsack
        # indiv = (cromo, fitness)
        indiv = copy.deepcopy(cromo)
        capacity = problem['capacity']
        pheno = phenotype(indiv)
        # sort by value
        pheno.sort(key=operator.itemgetter(2))
        
        weight_indiv = get_weight(indiv, problem)
        for index, weight, value in pheno:
            if weight_indiv <= capacity:
                break
            indiv[index] = 0
            weight_indiv -= weight
        return indiv
    
    def repair_weight(cromo):
        # Returns a valid individual by removing the heaviest item from the knapsack
        indiv = copy.deepcopy(cromo)
        capacity = problem['capacity']
        pheno = phenotype(indiv)
        pheno.sort(key=operator.itemgetter(1))
        
        weight_indiv = get_weight(indiv, problem)
        for index, weight, value in pheno:
            if weight_indiv <= capacity:
                break
            indiv[index] = 0
            weight_indiv -= weight
        return indiv
    
    def repair_value_to_profit(cromo):
        # Returns a valid individual by removing the least ratio value/weight item from the knapsack
        indiv = copy.deepcopy(cromo)
        capacity = problem['capacity']
        pheno = phenotype(indiv)
        pheno = [[i, w, v, float(v/w)] for i, w, v in pheno]
        pheno.sort(key= operator.itemgetter(3))
        
        weight_indiv = get_weight(indiv, problem)
        for index, weight, value, ratio in pheno:
            if weight_indiv <= capacity:
                break
            indiv[index] = 0
            weight_indiv -= weight
        return indiv