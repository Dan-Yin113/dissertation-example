'''

The reference of this code is https://github.com/jbaayen/homotopy-example

'''
import time
import casadi as ca
import numpy as np
import math

import matplotlib.pyplot as plt
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import ListedColormap, Normalize

from bokeh.layouts import column, gridplot
from bokeh.models import ColumnDataSource, CustomJS, Slider
from bokeh.plotting import figure, output_file, show

# These parameters correspond to Table 1
T = 72
dt = 600
times = np.arange(0, (T + 1) * dt, dt)

l = 10000 #the pipe length
n_level_nodes = 10 

lam = 0.0001#The friction coefficient of pipe

D = 0.5#The Diameter
a=D ** 2 * math.pi / 4#The cross-section area


h=0 #The slope of pipe


# Generic constants showed in Table 2
g=9.81
C=340
eps = 1e-12


#
n_theta_steps = 10
trace_path = True


# Derived quantities
dx = l / n_level_nodes


# Smoothed absolute value function
sabs = lambda x: ca.sqrt(x ** 2 + eps)


# Compute steady state initial condition
Q0 = np.full(n_level_nodes, 27.78)
P0 = np.full(n_level_nodes, 200000)# 200000 kgf/m^2 = 20 bar
# minus friction term
for i in range(1, n_level_nodes):
    P0[i] = P0[i - 1] - (lam * (C ** 2) / (2 * D * a)) * ((Q0[i-1]*sabs(Q0[i-1]))/P0[i-1]) 


# Symbols
Q = ca.MX.sym("Q", n_level_nodes, T)
P = ca.MX.sym("P", n_level_nodes, T)
theta = ca.MX.sym("theta")


# Left boundary condition
Q_left = np.full(T + 1, 27.78)
Q_left[T // 3 : 2 * T // 3] = 83.33
Q_left = ca.DM(Q_left).T


Q_full = ca.vertcat(Q_left, ca.horzcat(Q0, Q))
P_full = ca.horzcat(P0, P)


#Constraints(discretization)
c=(P_full[:, 1:] - P_full[:, :-1]) / dt + (C**2 / a) * (Q_full[1:, 1:] - Q_full[:-1, 1:]) / dx
     
d = (
     (Q_full[1:-1, 1:] - Q_full[1:-1, :-1]) / dt + a * (P_full[1:, 1:] - P_full[:-1, 1:]) / dx
     + lam / (2 * D) * (
        theta * (C**2 / a) *(Q_full[1:-1, 1:] * sabs(Q_full[1:-1, 1:]) / P_full[:-1, 1:])                    
        + (1 - theta) * sabs(Q_full[1:-1, 1:]) * 4.2 
                       )                   
    + g * a / C**2 * h * P_full[:-1, 1:] 
)                
     


# Objective function(maximize the outflow)
f = -ca.sum1(Q[-1])





# Variable bounds
lbQ = np.full(n_level_nodes, -305.56)
lbQ[-1] = 27.78
ubQ = np.full(n_level_nodes, 305.56)
ubQ[-1] = 111.11

lbQ = ca.repmat(ca.DM(lbQ), 1, T)
ubQ = ca.repmat(ca.DM(ubQ), 1, T)
lbP = ca.repmat(-np.inf, n_level_nodes, T)
ubP = ca.repmat(np.inf, n_level_nodes, T)

# Optimization problem
assert Q.size() == lbQ.size()
assert Q.size() == ubQ.size()
assert P.size() == lbP.size()
assert P.size() == ubP.size()

X = ca.veccat(Q, P)
lbX = ca.veccat(lbQ, lbP)
ubX = ca.veccat(ubQ, ubP)

g = ca.veccat(c, d)
lbg = ca.repmat(0, g.size1())
ubg = lbg

nlp = {"f": f, "g": g, "x": X, "p": theta}
solver = ca.nlpsol(
    "nlpsol",
    "ipopt",
    nlp,
    {
        "ipopt": {
            "tol": 1e-5,
            "constr_viol_tol": 1e-5,
            "acceptable_tol": 1e-5,
            "acceptable_constr_viol_tol": 1e-5,
            "print_level": 0,
            "print_timing_statistics": "no",
            "fixed_variable_treatment": "make_constraint",
        }
    },
)

# Initial guess
x0 = ca.repmat(30, X.size1())


# Solve
t0 = time.time()

results = {}

def solve(theta_value, x0):
    solution = solver(lbx=lbX, ubx=ubX, lbg=lbg, ubg=ubg, p=theta_value, x0=x0)
    if solver.stats()["return_status"] != "Solve_Succeeded":
        raise Exception(
            "Solve failed with status {}".format(solver.stats()["return_status"])
        )
    x = solution["x"]
    Q_res = ca.reshape(x[: Q.size1() * Q.size2()], Q.size1(), Q.size2())
    P_res = ca.reshape(x[Q.size1() * Q.size2() :], P.size1(), P.size2()) 
    d = {}
    d["Q_0"] = np.array(Q_left).flatten()
    for i in range(n_level_nodes):
        d[f"Q_{i + 1}"] = np.array(ca.horzcat(Q0[i], Q_res[i, :])).flatten()
        d[f"P_{i + 1}"] = np.array(ca.horzcat(P0[i], P_res[i, :])).flatten()
    results[theta_value] = d
    return x

if trace_path:
    for theta_value in np.linspace(0.0, 1.0, n_theta_steps):
        x0 = solve(theta_value, x0)
else:
    solve(1.0, x0)

print("Time elapsed in solver: {}s".format(time.time() - t0))





#plot in pdf
theta_values = list(results.keys())
variable_names = results[theta_values[0]].keys()

# Use greyscale style for plots
plt.style.use("grayscale")

file_type = "pdf"

# Generate Aggregated Plot
n_subplots = 2
width = 4
height = 4
time_hrs = times / 3600
fig, axarr = plt.subplots(n_subplots, sharex=True, figsize=(width, height))
theta = 1.0
vars_to_plot = "P_1", "P_4", "P_7", "P_10", "Q_0", "Q_3", "Q_7", "Q_10"
for var in vars_to_plot:
    if var == "Q_0":
        axarr[0].step(
            time_hrs,
            results[theta][var],
            where="mid",
            label=f"${var.split('_')[0]}_{{{var.split('_')[1]}}}$",
        )
        continue
    axarr[0 if var.startswith("Q") else 1].plot(
        time_hrs,
        results[theta][var],
        label=f"${var.split('_')[0]}_{{{var.split('_')[1]}}}$",
    )

axarr[0].set_ylabel("Flow Rate [m^3/s]")
axarr[1].set_ylabel("pressure [kgf/m^2]")
axarr[1].set_xlabel("Time [hrs]")

# Shrink margins
plt.autoscale(enable=True, axis="x", tight=True)
fig.tight_layout()

# Shrink each axis and put a legend to the right of the axis
for i in range(n_subplots):
    box = axarr[i].get_position()
    axarr[i].set_position([box.x0, box.y0, box.width * 0.85, box.height])
    axarr[i].legend(
        loc="center left", bbox_to_anchor=(1, 0.5), frameon=False, prop={"size": 8}
    )

# Output Plot
plt.savefig(f"final_results.{file_type}")



