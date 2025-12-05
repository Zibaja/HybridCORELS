import numpy as np
import json
from scipy.stats import ttest_rel, wilcoxon, kruskal
import scikit_posthocs as sp
import pandas as pd
from utils import *



class StatisticalAnalysis():
    """
    This class does a statistical analysis for a split of data (either test or train) that includes transpareny
    rates for each item in a given subgroups
    """
    def __init__(self, subgroups, data, method, split):
        self.subgroups = subgroups
        self.num_of_groups = len(self.subgroups )
        self.method = method
        self.split = split
        self.data = {k:data[method][split][k] for k in self.subgroups} #slice the data based on subgroups
        


    def info_subgroups (self):
        mean_subgroup = {k:np.mean(self.data[k]) for k in self.subgroups}
        std_subgroup = {k:np.std(self.data[k]) for k in self.subgroups}
        return mean_subgroup, std_subgroup

    
    def paired_ttest(self):
        if self.num_of_groups !=2:
            raise ValueError('paired t test is only done for two subgroups')
        arr1 = np.array(self.data[self.subgroups[0]])
        arr2 = np.array(self.data[self.subgroups[1]])
        arr_dif = arr1 - arr2
        mean_dif = np.mean(arr_dif)
        t_stat, p_val = ttest_rel(arr1, arr2)
        return mean_dif, t_stat, p_val


    def Wilcoxon(self):
        if len(self.subgroups)!=2:
            raise ValueError('Wilcoxon test is only done for two subgroups')
        arr1 = np.array(self.data[self.subgroups[0]])
        arr2 = np.array(self.data[self.subgroups[1]])
        arr_dif = arr1 - arr2
        mean_dif = np.mean(arr_dif)
        w_stat, p_val = wilcoxon(arr1, arr2)
        return mean_dif, w_stat, p_val
    
    def Kruskal(self):
        stat, p_val = kruskal(*self.data.values())
        return stat, p_val 
    

    def posthoc_dunn (self):
        df = pd.DataFrame({
            "value": np.concatenate(list(self.data.values())).astype(float),
            "group": sum([[name]*len(vals) 
                        for name, vals in self.data.items()], [])
        }) 
        result = sp.posthoc_dunn(df, val_col='value', group_col='group', p_adjust='bonferroni')
        return result 


    def effect_size(self,test_type, Kruskal_stat=None):
        total_observations = sum([len(i) for i in self.data.values()])
        if test_type == 'cohen':
            if len(self.subgroups)!=2:
                raise ValueError('paired t test is only done for two subgroups')
            d = (np.mean(np.array(self.data[self.subgroups[0]])) - np.mean(np.array(self.data[self.subgroups[1]])) ) / \
            np.sqrt((np.array(self.data[self.subgroups[0]]).std(ddof=1)**2 + np.array(self.data[self.subgroups[1]]).std(ddof=1)**2) / 2)
            if d>1.2 :
                print (f"Large effect size {d:.3f}")
            else:
                print(f"Effect size {d:.3f} is not large")
            return d
        
        if test_type == 'eta_squared':
            eta_squared = (Kruskal_stat - self.num_of_groups + 1) / (total_observations - self.num_of_groups)
            if eta_squared > 0.50 :
                print (f"Very large effect size, eta_squared= {eta_squared:.3f}")
            elif eta_squared > 0.14:
                print(f"The effect size of eta_squared {eta_squared:.3f} is large ")
            else:
                print(f'the effect size of eta_squared {eta_squared:.3f} is not large')
            return eta_squared
       
        if test_type == 'epsilon_squared':
            epsilon_squared = Kruskal_stat / (total_observations - 1)
            if epsilon_squared > 0.50 :
                print (f"Very large effect size, epsilon_squared= {epsilon_squared:.3f}")
            elif epsilon_squared > 0.14:
                print(f"The effect size of epsilon_squared {epsilon_squared:.3f} is large ")
            else:
                print(f'the effect size of epsilon_squared {epsilon_squared:.3f} is not large')
            return epsilon_squared
#         # η² ~ 0.01 → small effect
#         # η² ~ 0.06 → medium effect
#         # η² > 0.14 → large effect
#         # η² > 0.50 → VERY large subgroup disparity
#         # Same interpretation for ε².

    


def main():

    Method = 'HybridCORELSPostClassifier'  # Supported: "HybridCORELSPreClassifier", "HybridCORELSPostClassifier"

    data = read_json(Path (Data_dir/'fairness_0.8_HybridCORELSPostClassifier.json'))

    condition_gender = ['Gender=Male', 'neg_Gender=Male']
    condition_race = ['Race=African-American', 'Race=Caucasian', 'Race=Hispanic', 'Race=Other'] # all race subgropus in compas
    condition_age = ['Age=18-25','Age=26-29','Age>=30']

    condition = condition_age # this is the condition that will be applied in statistical analysis

    alpha = 0.05
    stat_test = StatisticalAnalysis(subgroups= condition, data=data, method = Method , split='test')
    print('statistical information is as follows:')
    print(f"Mean of each subgroups :{stat_test.info_subgroups()[0]}")
    print(f"Standard deviation of each subgroups :{stat_test.info_subgroups()[1]}")
    if stat_test.num_of_groups == 2:
        mean_dif, _, p_val = stat_test.paired_ttest()
        if p_val < alpha:
            print(f"Based on {stat_test.paired_ttest.__name__} subgroup transparency rates are statistically significantly different")
        else:
            print(f'Based on {stat_test.paired_ttest.__name__} there is no evidence to approve difference')
        mean_dif, _, p_val = stat_test.Wilcoxon()
        
        if p_val < alpha:
            print(f"Based on {stat_test.Wilcoxon.__name__} subgroup transparency rates are statistically significantly different")
        else:
            print(f'Based on {stat_test.Wilcoxon.__name__} there is no evidence to approve difference')

        stat_test.effect_size(test_type='cohen')
    elif stat_test.num_of_groups>2 :
        H, P = stat_test.Kruskal()
        if P < alpha:
            print(f"'Based on {stat_test.Kruskal.__name__} at least one subgroup differs significantly from the others.")
        else :
            print(f"Based on {stat_test.Kruskal.__name__} no subgroup differs significantly from the others")
        result = stat_test.posthoc_dunn()
        print(result)
        stat_test.effect_size(test_type='eta_squared', Kruskal_stat= H )
        stat_test.effect_size(test_type='epsilon_squared', Kruskal_stat= H )







if __name__ == '__main__':
    main()

