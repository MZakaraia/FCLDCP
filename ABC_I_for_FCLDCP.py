import numpy as np
import HeuristicApproach as ha
import copy


def chaotic_mapping_vector(dim):
    vector = np.array([])
    h = np.random.uniform()
    gamma = h
    for _ in range(dim):
        vector = np.append(vector,gamma)
        h = 4 * h*(1 - h)
        gamma = 2/np.pi * np.arcsin(np.sqrt(h))
    return vector

#  Initialization Phase
def Initialize(popSize , Demands, Capacities, membershipvalue):
    population = []
    for _ in range(popSize):
        sol = ha.solStucture(Demands, Capacities, membershipvalue)
        sol.DistCenterArray = chaotic_mapping_vector(sol.NumberOFDistCenter)
        PlantArray = chaotic_mapping_vector(sol.NumberOfPlants)
        population.append(sol)
    return population

def sphere(x):
    return sum(x**2)

def evaluationFunc(population, TransCosts, FixedCosts):
    Evals = np.array([i.Evaluate(TransCosts, FixedCosts) for i in population])
    bestBee = population[np.argmin(Evals)]
    return np.min(Evals), bestBee, Evals


#  Employed Bees Phase
def EmployedBeesFun(population, popSize, abandonmentVector, abandonmentLimit, Evals, BestEval, bestBee\
    , Demands, Capacities, TransCosts, FixedCosts, Itr, MaxItr):
    k = 0
    for i in population:
        randchoice = np.random.choice([j for j in range(popSize) if j != k], 1)[0]
        rndBee = population[randchoice]
        rank = np.where(k == np.argsort(Evals))[0][0]
        mu = rank/popSize
        cValue = cvaluefn(Itr, MaxItr, mu)
        Employed = copy.deepcopy(i)
        Employed.DistCenterArray = Employed.DistCenterArray + cValue *(Employed.DistCenterArray - rndBee.DistCenterArray) + \
            (1- cValue) * (bestBee.DistCenterArray - Employed.DistCenterArray)
        Employed.PlantArray = Employed.PlantArray + cValue *(Employed.PlantArray - rndBee.PlantArray) + \
            (1- cValue) * (bestBee.PlantArray - Employed.PlantArray)  
        EmployedEval = Employed.Evaluate(TransCosts, FixedCosts)
        if EmployedEval <= Evals[k]:
            population[k] = copy.deepcopy(Employed)
            Evals[k] = EmployedEval
        if EmployedEval <= BestEval:
            BestEval = EmployedEval
            bestBee = copy.deepcopy(Employed)

        # Scout Bees Phase
        else:
            abandonmentVector[k] += 1
            if abandonmentVector[k] == abandonmentLimit:
                population[k] = ha.solStucture(Demands, Capacities)
                Evals[k] = population[k].Evaluate(TransCosts, FixedCosts)
                abandonmentVector[k] = 0
        k += 1
    return population, Evals, BestEval, bestBee

def CalculateFitness(Evals):
    fitness = []
    for i in Evals:
        if i >= 0:
            fitness.append(1/(1+i))
        else:
            fitness.append(1+abs(i))
    return fitness
Omega = lambda Itr, MaxItr:  0.1 + (0.9 - 0.1) * (Itr / MaxItr)
cvaluefn = lambda Itr, MaxItr, mu:np.sin(2 * mu ** Omega(Itr, MaxItr)/np.pi)
SelectionProb = lambda ranks, om, popsize: om * (1-om)**(ranks-1)/(1-(1-om)**popsize)
# Onlooker Bees Phase
def OnlookerBees(population, popSize, abandonmentVector, abandonmentLimit, Evals, BestEval, bestBee\
    , Demands, Capacities, TransCosts, FixedCosts, Itr, MaxItr):
    k = 0
    for i in population:
        Onlooker = copy.deepcopy(i)
        # EvalsSum = sum(Evals)
        # fitnessProb = np.array([i/EvalsSum for i in Evals])
        ranks = np.argsort(Evals)
        rank = np.where(k == np.argsort(Evals))[0][0]
        mu = rank/popSize
        cValue = cvaluefn(Itr, MaxItr, mu)
        Om = Omega(Itr, MaxItr)
        SelectionProbability = SelectionProb(ranks, Om, popSize)
        SelectionProbSum = SelectionProbability.sum()
        fitnessProb = [j/SelectionProbSum for j in SelectionProbability]
        Selected = population[np.random.choice(range(popSize), p = fitnessProb)]
        Onlooker.DistCenterArray = Onlooker.DistCenterArray + cValue\
            *(Onlooker.DistCenterArray - Selected.DistCenterArray) + (1- cValue)*(bestBee.DistCenterArray - Onlooker.DistCenterArray)
        Onlooker.PlantArray = bestBee.PlantArray + cValue*(Onlooker.PlantArray - Selected.PlantArray) + \
            (1- cValue)*(bestBee.PlantArray - Onlooker.PlantArray)  
        OnlookerEval = Onlooker.Evaluate(TransCosts, FixedCosts)
        if OnlookerEval <= Evals[k]:
            population[k] = copy.deepcopy(Onlooker)
            Evals[k] = OnlookerEval
        if OnlookerEval <= BestEval:
            BestEval = OnlookerEval
            bestBee = copy.deepcopy(Onlooker)
        # Scout Bees Phase
        else:
            abandonmentVector[k] += 1
            if abandonmentVector[k] == abandonmentLimit:
                population[k] = ha.solStucture(Demands, Capacities)
                Evals[k] = population[k].Evaluate(TransCosts, FixedCosts)
                abandonmentVector[k] = 0
        k += 1
    return population, Evals, BestEval, bestBee

def ABC_Algorithm_FCLDCP_I(popSize, MaxItr, abandonmentLimit, Demands, Capacities, TransCosts, FixedCosts, membershipvalue = 0.6):
    abandonmentVector = np.zeros(popSize)
    population = Initialize(popSize , Demands, Capacities, membershipvalue)
    # global BestEval, bestBee    
    BestEval, bestBee, Evals = evaluationFunc(population, TransCosts, FixedCosts)
    # Evals = [sphere(i) for i in population]
    # BestEvalCopy = copy.deepcopy(BestEval)
    ConvergenceVector = [BestEval]
    for Itr in range(MaxItr):
        # print(BestEval)
        # Employed Bees Phase
        # population, Evals = EmployedBeesFun(population, a, popSize, abandonmentVector, abandonmentLimit, dim, Evals)
        population, Evals, BestEval, bestBee = EmployedBeesFun(population, popSize, abandonmentVector, abandonmentLimit, Evals, BestEval, bestBee\
                , Demands, Capacities, TransCosts, FixedCosts, Itr, MaxItr)
        # Onlooker Bees Phase 
        # population, Evals = OnlookerBees(population, a, popSize, abandonmentVector, abandonmentLimit, dim, Evals)
        population, Evals, BestEval, bestBee = OnlookerBees(population, popSize, abandonmentVector, abandonmentLimit, Evals, BestEval, bestBee\
                , Demands, Capacities, TransCosts, FixedCosts, Itr, MaxItr)
        # BestEvalCopy = copy.deepcopy(BestEval)
        ConvergenceVector.append(BestEval)
    
    return bestBee, ConvergenceVector