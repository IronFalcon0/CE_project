

class Knapsack():
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

    def merito(problem):
        def fitness(indiv):
            quali = evaluate_zero(phenotype(indiv))
            return quali
        return fitness
    
    def phenotype(indiv):
        pheno = [[id, problem['weights'][id], problem['values'][id]] for id in range(len(indiv)) if indiv[id] == 1]
        return pheno
    
    def evaluate_zero(feno):
        total_weight = sum([weight for id,weight,value in feno])
        if total_weight > problem['capacity']:
            return 0
        return sum([value for id,weight,value in feno])
    
    def generate_uncor(size_items, max_value):
        weights = [random.uniform(1, max_value) for i in range(size_items)]
        values = [random.uniform(1, max_value) for i in range(size_items)]
        capacity = int(0.5 * sum(weights))
        return {'weights': weights, 'values': values, 'capacity': capacity}
    
    def generate_weak_cor(size_items, max_value, amplitude):
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
        weight = [random.uniform(1, max_value) for i in range(size_items)]
        values = [weights[i] + amplitude for i in range(size_items)]
        capacity = int(0.5 * sum(weights))
        return {'weights': weights, 'values': values, 'capacity': capacity}\
    
    def evaluate_log(feno):
        total_weight = sum([weight for id,weight,value in feno])
        quality = sum([value for id,weight,value in feno])
        capacity = problem['capacity']
        if total_weight > capacity:
            pho = max([v/w for i,w,v in feno])
            quality -= math.log((total_weight - capacity) * pho + 1,2)
        return quality
    
    def evaluate_linear(feno):
        total_weight = sum([weight for id,weight,value in feno])
        quality = sum([value for id,weight,value in feno])
        capacity = problem['capacity']
        if total_weight > capacity:
            pho = max([v/w for i,w,v in feno])
            quality -= (total_weight - capacity) * pho
        return quality
    
    def evaluate_quadratic(feno):
        total_weight = sum([weight for id,weight,value in feno])
        quality = sum([value for id,weight,value in feno])
        capacity = problem['capacity']
        if total_weight > capacity:
            pho = max([v/w for i,w,v in feno])
            quality -= (total_weight - capacity) ** 2 * pho
        return quality
    
    def repair_value(cromo):
        indiv = copy.deepcopy(cromo)
        capacity = problem['capacity']
        pheno = phenotype(indiv)
        pheno.sort(key=operator.itemgetter(2))
        
        weight_indiv = get_weight(indiv, problem)
        for index, weight, value in pheno:
            if weight_indiv <= capacity:
                break
            indiv[index] = 0
            weight_indiv -= weight
        return indiv
    
    def repair_weight(cromo):
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
    
    def list_items(problem):
        weight_value_list = list(zip(problem['weights'], problem['values']))
        l_items = [[i, w, v, v/w] for i, (w, v) in enumerate(weight_value_list)]
        return l_items
    
    def decode_int(indiv, problem):
        capacity = problem['capacity']
        l_items = list_items(problem)
        sum_weight = 0
        res = []
        for i in range(len(indiv)):
            j = indiv[i]
            id, w, v, r = l_items.pop(j)
            j = indiv[i]
            i, w, v, r = l_items.pop(j)
            if sum_weight + w <= capacity:
                sum_weights += w
                res.append([id, w, v, r])
            else:
                return res
        return res
    
    
    def run(self):
        pass