# python 3
# this class combines all basic features of a generic player
#import pandas as pd
import numpy as np
#import matplotlib.pyplot as pl
import pulp


class Player:

    def __init__(self):
        # some player might not have parameters
        self.parameters = 0
        self.horizon = 48
        self.res= np.array([])
        
    def set_scenario(self, scenario_data):
        self.data = scenario_data   # il faut que ce soit une liste qui contient la conso de l'usine à chaque pas de temps

    def set_prices(self, prices):
        self.prices = prices

    def compute_all_load(self):
        load = np.zeros(self.horizon)
        for time in range(self.horizon):
            load[time] = self.compute_load(time)
        return load

    def take_decision(self, time):
        if self.res.size == 0 :
            self.optimize()
        battery_load = self.res[time]
        return battery_load

    def compute_load(self, time):
        load = self.take_decision(time)
        # do stuff ?
        load += self.data[time]
        return load

    def reset(self):
        self.res = np.array([])
        

    def optimize(self):
        Delta_t = 0.5
        rho = 0.95
        my_lp_problem = pulp.LpProblem("My_LP_Problem", pulp.LpMinimize)
        variables = {}
        for t in range(48):
            variables[t] = {}
            var_name = "battery_load" + str(t)
            variables[t]["battery_load"] = pulp.LpVariable(var_name, -10, 10)  # en kW

            var_name = "battery_load_plus" + str(t)
            variables[t]["battery_load_plus"] = pulp.LpVariable(var_name, 0)

            var_name = "total_load" + str(t)
            variables[t]["total_load"] = pulp.LpVariable(var_name)

            var_name = "battery_stock" + str(t)
            variables[t]["battery_stock"] = pulp.LpVariable(var_name, 0, 60)  # en kW

            var_name = "load_plus" + str(t)
            variables[t]["load_plus"] = pulp.LpVariable(var_name,0)

            constraint_name = "egalite_centrale" + str(t)
            my_lp_problem += variables[t]["total_load"] == self.data[t] + variables[t]["battery_load"], constraint_name # être sûr de la forme de data !!!!!!!


            constraint_name = "lift2" + str(t)
            my_lp_problem += variables[t]["battery_load_plus"] >= variables[t]["battery_load"], constraint_name

            constraint_name = "lift_load" + str(t)
            my_lp_problem += variables[t]["load_plus"] >= variables[t]["total_load"] , constraint_name

            if t == 0:
                constraint_name = "init_stock"
                my_lp_problem += variables[t]["battery_stock"] == variables[t]["battery_load"]*Delta_t , constraint_name
            else:
                constraint_name = "stock" + str(t)
                my_lp_problem += variables[t]["battery_stock"] == variables[t - 1]["battery_stock"] + \
                                 ((1. / rho) * variables[t]["battery_load"] + (rho - (1. / rho)) *
                                  variables[t]["battery_load_plus"]) * Delta_t , constraint_name



        constraint_name = "final_stock"
        my_lp_problem += variables[47]["battery_stock"] == 0, constraint_name

        constraint_name = "init_battery_load"
        my_lp_problem += variables[0]["battery_load"] >= 0, constraint_name

        my_lp_problem.setObjective(
            pulp.lpSum([self.prices["sell"][t] * variables[t]["total_load"]*Delta_t +
                         (self.prices["purchase"][t]- self.prices["sell"][t])*variables[t]["load_plus"]* Delta_t for t in range(48)]))
        my_lp_problem.solve()

        res = []
        for t in range(48):
            for variable in my_lp_problem.variables():
                if variable.name == "battery_load" + str(t):
                    res.append(variable.varValue)
        self.res = np.array(res)
"""
A = pd.read_csv('indus_cons_scenarios.csv')
B = A[A["site_id"]==1]
days = []
for i in range(1,31):
    C = B[B["scenario"]==i]
    days.append(C.to_numpy()[:,3])

P = Player()
P.set_prices(100*np.random.rand(48))
P.set_scenario(days[0])
print(P.compute_all_load())
"""
