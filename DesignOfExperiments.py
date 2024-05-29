import HeuristicApproach as ha
from ABC_for_FCLDCP import *
import matplotlib.pyplot as plt
import pandas as pd
from ABC_I_for_FCLDCP import *
from scipy.stats import ttest_ind
from scipy.stats import wilcoxon
from scipy.stats import friedmanchisquare

PopsizeLevels = [40, 60, 80, 100]
MaxItrLevels = [40, 60, 80, 100]
abandonment = [10, 20, 30, 40]
ScalingFactor = [0.25, 0.50, 0.75, 1]

# Parameters = {"Popsize": PopsizeLevels, "MaxItr": MaxItrLevels, "Abandonment":Abandonment}

ABC_M = pd.read_excel('ABC_M_Taguchi.xlsx')


class TaguchiAnalysis:
    def __init__(self, ResponseFile, **param):
        self.param = param
        self.response = ResponseFile
        self.NumOfParameters = len(param)
    
    def PlotMainEffects(self):       
        k = 0
        for i in self.param:            
            SNValues = np.array([], dtype=float)
            for j in self.param[i]:
                Values = self.response[self.response.iloc[:,k] == j].iloc[:,-1]
                SN = 10 * np.log(Values.mean() ** 2/ Values.std() ** 2)
                SNValues = np.append(SNValues, SN)
            k += 1
            plt.subplot(1, self.NumOfParameters, k)
            plt.plot([str(j)  for j in self.param[i]], SNValues, '-o')
            plt.title(i)
        plt.show()



PopsizeLevels = [40, 60, 80, 100]
MaxItrLevels = [40, 60, 80, 100]
abandonment = [10, 20, 30, 40]
ScalingFactor = [0.25, 0.50, 0.75, 1]
ABC_M = pd.read_excel('ABC_M_Taguchi.xlsx')
Experiment = TaguchiAnalysis(ABC_M, Popsize = PopsizeLevels, MaxItr = MaxItrLevels, Abandonment = abandonment, Scaling_Factor = ScalingFactor)
# Experiment = TaguchiAnalysis(ABC_M, Popsize = PopsizeLevels, MaxItr = MaxItrLevels, Abandonment = abandonment)
Experiment.PlotMainEffects()

PopsizeLevels = [40, 60, 80, 100]
MaxItrLevels = [40, 60, 80, 100]
abandonment = [10, 20, 30, 40]
ScalingFactor = [0.25, 0.50, 0.75, 1]
ABC_M = pd.read_excel('ABC_I_Taguchi.xlsx')
Experiment = TaguchiAnalysis(ABC_M, Popsize = PopsizeLevels, MaxItr = MaxItrLevels, Abandonment = abandonment)
# Experiment = TaguchiAnalysis(ABC_M, Popsize = PopsizeLevels, MaxItr = MaxItrLevels, Abandonment = abandonment)
Experiment.PlotMainEffects()

PopsizeLevels = [50, 80, 100, 150]
MaxItrLevels = [50, 80, 100, 150]
ArchiveSize = [10, 20, 30, 40]
ScalingFactor = [0.25, 0.50, 0.75, 1]
ABC_M = pd.read_excel('ABC_II_Taguchi.xlsx')
Experiment = TaguchiAnalysis(ABC_M, Popsize = PopsizeLevels, MaxItr = MaxItrLevels, ArchiveSize = ArchiveSize)
# Experiment = TaguchiAnalysis(ABC_M, Popsize = PopsizeLevels, MaxItr = MaxItrLevels, Abandonment = abandonment)
Experiment.PlotMainEffects()


