########################################################################################
# Workshop: Coding for PDEs with Neural Networks
# Date: 2025-24-01
# Author: Danilo Aballay, Vicente Iligaray, Ignacio Tapia y Manuel Sánchez
########################################################################################

import jax
import jax.numpy as jnp
from jax import jit

global problem_number
problem_number=2

# Element stiffness matrix and load vector
def element_stiffness(h):
    return jnp.array([[1, -1], [-1, 1]], dtype=jnp.float64) / h

def element_load(coords):
    x1, x2 = coords
    p1 = -1/jnp.sqrt(3)
    p2 = 1/jnp.sqrt(3)
    pt1 = (x2 - x1) * p1 / 2 + (x2 + x1) / 2
    pt2 = (x2 - x1) * p2 / 2 + (x2 + x1) / 2
    phiatpt1 = (p2+1)/2
    phiatpt2 = (1+p1)/2
    #midpoint = (x1 + x2) / 2
    h = x2 - x1
    problem_test = problem(problem_number)
    f = problem_test.f
    return h * jnp.array([f(pt1)*phiatpt1 + f(pt2)*phiatpt2, f(pt1)*phiatpt2 + f(pt2)*phiatpt1], dtype=jnp.float64) / 2

# Assemble global stiffness matrix and load vector
def assemble(n_elements, node_coords, element_length, n_nodes):
    element_nodes = jnp.array([[i, i + 1] for i in range(n_elements)])
    coords = node_coords[element_nodes]
    h_values = element_length

    ke_values = jax.vmap(element_stiffness)(h_values)
    fe_values = jax.vmap(element_load)(coords)

    K = jnp.zeros((n_nodes, n_nodes))
    F = jnp.zeros(n_nodes)

    K = K.at[element_nodes[:, 0], element_nodes[:, 0]].add(ke_values[:, 0, 0])
    K = K.at[element_nodes[:, 0], element_nodes[:, 1]].add(ke_values[:, 0, 1])
    K = K.at[element_nodes[:, 1], element_nodes[:, 0]].add(ke_values[:, 1, 0])
    K = K.at[element_nodes[:, 1], element_nodes[:, 1]].add(ke_values[:, 1, 1])

    F = F.at[element_nodes[:, 0]].add(fe_values[:, 0])
    F = F.at[element_nodes[:, 1]].add(fe_values[:, 1])

    return K, F

# Apply boundary conditions
def apply_boundary_conditions(K, F):
    problem_test = problem(problem_number)
    bc_g0 = problem_test.g0
    bc_g1 = problem_test.g1
    # bc_g0 = g0()
    # bc_g1 = g1()

    F = F - K[:, 0] * bc_g0
    # F = F - K[:, -1] * bc_g1

    K = K.at[0, :].set(0)
    K = K.at[:, 0].set(0)
    # K = K.at[-1, :].set(0)
    # K = K.at[:, -1].set(0)
    K = K.at[0, 0].set(1)
    # K = K.at[-1, -1].set(1)

    F = F.at[0].set(bc_g0)
    # F = F.at[-1].set(bc_g1)
    F = F.at[-1].set(F[-1]+0.7)

    return K, F

# Solve the system
def solve(nelem):
    problem_test = problem(problem_number)
    bc_g0 = problem_test.g0
    bc_g1 = problem_test.g1

    n_nodes = nelem + 1
    n_elements = nelem
    node_coords = jnp.linspace(0, 1, n_nodes)
    element_length = node_coords[1:] - node_coords[:-1]

    K, F = assemble(n_elements, node_coords, element_length, n_nodes)
    K, F = apply_boundary_conditions(K, F)
    # u = jnp.linalg.solve(K, F) - (bc_g0*(1-node_coords) + bc_g1*(node_coords-0))
    u = jnp.linalg.solve(K, F)
    loss = 0.5*jnp.dot(u, jnp.dot(K, u)) - jnp.dot(F, u)

    return loss

# Solve the system
def solve_array(node_coords):
    problem_test = problem(problem_number)
    bc_g0 = problem_test.g0
    bc_g1 = problem_test.g1

    n_nodes = node_coords.size 
    n_elements = n_nodes - 1
    # node_coords = jnp.linspace(0, 1, n_nodes)
    element_length = node_coords[1:] - node_coords[:-1]

    K, F = assemble(n_elements, node_coords, element_length, n_nodes)
    K, F = apply_boundary_conditions(K, F)
    # u = jnp.linalg.solve(K, F) - (bc_g0*(1-node_coords) + bc_g1*(node_coords-0))
    u = jnp.linalg.solve(K, F)
    loss = 0.5*jnp.dot(u, jnp.dot(K, u)) - jnp.dot(F, u)

    return loss

class Elliptic1D:
    def __init__(self, f, g0, g1, sigma, u=None):
        """
        Initializes the 1D elliptic problem.

        Parameters:
        - f (callable): Right-hand side function f(x).
        - g0 (float): Dirichlet boundary condition at x=0.
        - g1 (float): Dirichlet boundary condition at x=1.
        - sigma (callable): Coefficient function sigma(x).
        - u (callable, optional): Solution function (if known, default is None).
        """
        self.a = 0.0  # Left endpoint of the domain
        self.b = 1.0  # Right endpoint of the domain
        self.f = f
        self.g0 = g0
        self.g1 = g1
        self.sigma = sigma
        self.u = u  # Analytical solution, if provided

def problem(problem_number):
    """
    Returns an Elliptic1D problem instance based on the problem number.

    Parameters:
    - problem_number (int): The index of the desired problem (0-based).

    Returns:
    - Elliptic1D: An instance of the elliptic problem.
    """
    problems_data = [
        {
            "f": lambda x: 0 * x,  # Zero source term
            "g0": 0.5,  # Dirichlet boundary condition at x=0
            "g1": -0.5,  # Dirichlet boundary condition at x=1
            "sigma": lambda x: 1,  # Constant coefficient sigma(x)
        },
        {
            "f": lambda x: jnp.sin(jnp.pi * x),  # Sinusoidal source term
            "g0": 0,  # Dirichlet boundary condition at x=0
            "g1": 0,  # Dirichlet boundary condition at x=1
            "sigma": lambda x: 1,  # Constant coefficient sigma(x)
        },
        {
            "f": lambda x: 0.7 * 0.3 * x ** (-1.3),  # Singular source term
            "g0": 0,  # Dirichlet boundary condition at x=0
            "g1": 1,  # Dirichlet boundary condition at x=1
            "sigma": lambda x: 1,  # Constant coefficient sigma(x)
            "u": lambda x: x**(0.7),
        },
    ]

    # Ensure problem_number is within bounds
    if problem_number < 0 or problem_number >= len(problems_data):
        raise ValueError(f"Invalid problem number: {problem_number}. Must be between 0 and {len(problems_data) - 1}.")

    # Retrieve problem data and unpack into the Elliptic1D class
    data = problems_data[problem_number]
    return Elliptic1D(**data)


import matplotlib.pyplot as plt
import numpy as np
Nelem = 2**(jnp.arange(2,9))+1
FEMerr = []
rate = []
for i, nelem in enumerate(Nelem):
    # print(nelem)
    loss = solve(nelem)
    # print(loss)
    FEMerr.append(np.sqrt(2*jnp.abs(-0.6125-loss))*100/(0.49/0.4))
    if i>0:
        rate.append(np.log(FEMerr[-1]/FEMerr[-2])/np.log(Nelem[i-1]/Nelem[i]))

plt.loglog(Nelem,FEMerr,'o-')
plt.savefig('FEMlogN.png')
# print(FEMerr)
print(rate)

import csv
list_of_arrays = []
with open('save_coords.csv', 'r') as archive:
    reader = csv.reader(archive)

    for row in reader:
        list_of_arrays.append(list(map(float,row)))


NNerr = []
rate = []
for i, nelem in enumerate(Nelem):
    # print(nelem)

    # print(jnp.array(list_of_arrays[i]))
    coords = jnp.array(list_of_arrays[i])
    # print(Nelem[i], coords.size)
    loss = solve_array(coords)
    # print(loss)
    NNerr.append(np.sqrt(2*jnp.abs(-0.6125-loss))*100/(0.49/0.4))
    if i>0:
        rate.append(np.log(NNerr[-1]/NNerr[-2])/np.log(Nelem[i-1]/Nelem[i]))

plt.loglog(Nelem,NNerr,'o-')
plt.savefig('NNlogN.png')
print(rate)