# Solve the following MIP:
#  maximize
#        x +   y + 2 z
#  subject to
#        x + 2 y + 3 z <= 4
#        x +   y       >= 1
#        x, y, z binary

import gurobipy as gp
from gurobipy import *
import numpy as np
#TODO add the correct formules from paper.
if __name__=="__main__":
    # Create a new model
    m = Model("matrix_qcqp")

    # Create a 2D array of variables
    x = []
    for i in range(5):
        x.append([])
        for j in range(5):
            x[i].append(m.addVar(name="x_" + str(i) + "_" + str(j)))

    # Update model to integrate new variables
    m.update()

    # Set objective: sum of all x_ij^2
    obj = quicksum(x[i][j] * x[i][j] for i in range(5) for j in range(5))
    m.setObjective(obj, GRB.MINIMIZE)

    # Add constraint: sum of all x_ij >= 1
    m.addConstr(quicksum(x[i][j] for i in range(5) for j in range(5)) >= 1, "c0")

    # Add non-convex quadratic constraint: sum of all x_ij^2 <= 1
    m.addQConstr(quicksum(x[i][j] * x[i][j] for i in range(5) for j in range(5)) <= 1, "qc0")

    # Set to non-convex
    m.Params.NonConvex = 2

    # Optimize model
    m.optimize()
