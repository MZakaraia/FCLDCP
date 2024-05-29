import numpy as np
import HeuristicApproach as ha
import copy


#  Initialization Phase
def Initialize(popSize , Demands, Capacities, membershipvalue):
    population = [ha.solStucture(Demands, Capacities, membershipvalue) for _ in range(popSize)]
    return population



def evaluationFunc(population, TransCosts, FixedCosts):
    Evals = np.array([i.Evaluate(TransCosts, FixedCosts) for i in population])
    bestBee = population[np.argmin(Evals)]
    return np.min(Evals), bestBee, Evals


#  Employed Bees Phase
def EmployedBeesFun(population, popSize, CAEval, CA, Evals, BestEval, bestBee\
    , Demands, Capacities, TransCosts, FixedCosts, N):
    k = 0  
    for i in population:
        randchoice = np.random.choice([j for j in range(len(CA)) if j != k], 1)[0]
        rndBee = CA[randchoice]
        Employed = copy.deepcopy(i)
        Employed.DistCenterArray = rndBee.DistCenterArray + np.random.uniform(-1, 1)*(rndBee.DistCenterArray - Employed.DistCenterArray)
        Employed.PlantArray = rndBee.PlantArray + np.random.uniform(-1, 1)*(rndBee.PlantArray - Employed.PlantArray)  
        EmployedEval = Employed.Evaluate(TransCosts, FixedCosts)
        if EmployedEval < Evals[k]:
            population[k] = copy.deepcopy(Employed)
            Evals[k] = EmployedEval

            # Update convergence archive
            CA.append(Employed)            
            CAEval = np.append(CAEval, EmployedEval)

            if len(CAEval) >= N:                
                CA.pop(np.where(np.max(CAEval) == CAEval)[0][0])
                CAEval = np.delete(CAEval, np.where(np.max(CAEval) == CAEval)[0][0])


                
        elif EmployedEval == Evals[k]:
            if np.random.uniform() > 0.5:
                population[k] = Employed

        if EmployedEval <= BestEval:
            BestEval = EmployedEval
            bestBee = copy.deepcopy(Employed)



        k += 1
    return population, Evals, BestEval, bestBee, CAEval
# Scout Bees Phase
def scoutBess(alpha, N, TransCosts, FixedCosts, CAEval, CA, DAEval, DA):        
    for k in range(len(CA)):
        if np.random.uniform() < alpha:
            r1, r2, r3 = np.random.choice(range(len(CA)),3)
            CA[k].DistCenterArray = CA[r1].DistCenterArray + np.random.uniform(-1, 1) * (CA[r2].DistCenterArray - CA[r3].DistCenterArray)
            CA[k].PlantArray = CA[r1].PlantArray + np.random.uniform(-1, 1) * (CA[r2].PlantArray - CA[r3].PlantArray)
            CAEval[k] = CA[k].Evaluate(TransCosts, FixedCosts)
            # Update DA
            DA.extend([CA[k]])
            DAEval = np.insert(DAEval, len(DAEval), CAEval[k])
        DAindices = np.argsort(-DAEval)
        DA = [DA[i] for i in DAindices]
        DAEval = DAEval[DAindices]
        if len(DA) > N:
            DA = DA[:N]
            DAEval = DAEval[:N]
        
    


# Onlooker Bees Phase
def OnlookerBees(population, popSize, CAEval, CA, DA, Evals, BestEval, bestBee\
    , Demands, Capacities, TransCosts, FixedCosts, N):
    k = 0
    for i in population:        
        minimum = np.min(Evals)
        denominator = np.max(Evals) - minimum        
        fitnessProb = (i.Eval-minimum)/denominator        
        if np.random.uniform() < fitnessProb:
            if len(CA) >= 2:
                r1, r2 = np.random.choice(range(len(CA)), 2, replace=False)
                rand1 = CA[r1]
                rand2 = CA[r2]
            else:
                rand1 = CA[0]
                rand2 = CA[0]
            rand3 = DA[np.random.choice(range(len(DA)),1)[0]]

            Onlooker = copy.deepcopy(i)
            Onlooker.DistCenterArray = rand3.DistCenterArray + np.random.uniform(-1, 1)\
                *(rand1.DistCenterArray - rand2.DistCenterArray)
            Onlooker.PlantArray = rand3.PlantArray + np.random.uniform(-1, 1)*(rand1.PlantArray - rand2.PlantArray)  
            OnlookerEval = Onlooker.Evaluate(TransCosts, FixedCosts)
            if OnlookerEval <= Evals[k]:
                population[k] = copy.deepcopy(Onlooker)
                Evals[k] = OnlookerEval
            if OnlookerEval <= BestEval:
                BestEval = OnlookerEval
                bestBee = copy.deepcopy(Onlooker)
        k += 1
    return population, Evals, BestEval, bestBee

def ABC_II_Algorithm_FCLDCP(popSize, MaxItr, Demands, Capacities, TransCosts, FixedCosts, N = 20, membershipvalue = 0.6):
    population = Initialize(popSize , Demands, Capacities, membershipvalue)    
    BestEval, bestBee, Evals = evaluationFunc(population, TransCosts, FixedCosts)
    DAindices = np.argsort(Evals)
    CA = [population[DAindices[i]] for i in range(2)]
    CAEval = Evals[DAindices[:2]]
    DAEval = Evals[DAindices[2:]]
    DA = [population[DAindices[i]] for i in range(2,popSize)]    
    ConvergenceVector = [BestEval]
    for n in range(MaxItr):
        alpha = 1- n/MaxItr        
        population, Evals, BestEval, bestBee, CAEval = EmployedBeesFun(population, popSize, CAEval, CA, Evals, BestEval, bestBee\
                , Demands, Capacities, TransCosts, FixedCosts, N)
        population, Evals, BestEval, bestBee = OnlookerBees(population, popSize, CAEval, CA, DA,Evals, BestEval, bestBee\
                , Demands, Capacities, TransCosts, FixedCosts, N)
        scoutBess(alpha, N, TransCosts, FixedCosts, CAEval, CA, DAEval, DA)

        ConvergenceVector.append(BestEval)
    
    return bestBee, ConvergenceVector