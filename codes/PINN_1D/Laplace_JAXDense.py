########################################################################################
# Workshop: Coding for PDEs with Neural Networks
# Date: 2025-24-01
# Author: Danilo Aballay, Vicente Iligaray, Ignacio Tapia y Manuel Sánchez
########################################################################################

import jax
from jax import lax
import jax.numpy as jnp
from jax import jit
import keras

global problem_number
problem_number=3

def softmax_nodes(params):
    # Compute the softmax values
    softmax_values = jax.nn.softmax(params)
    # softmax_values = jax.nn.softmax(params)

    # print(paa)
    # Compute the cumulative sum of the softmax values
    cumulative_sum = jnp.cumsum(softmax_values)
    cumulative_sum_with_zero = jnp.insert(cumulative_sum, 0, 0)

    return cumulative_sum_with_zero

def element_stiffness(coords, SIGMA): # Discontinuous case
    aux1 = 2*jnp.sqrt(10/7)
    aux2 = 13*jnp.sqrt(70)
    nodes = jnp.array([-1/3*jnp.sqrt(5+aux1), -1/3*jnp.sqrt(5-aux1), 0, 1/3*jnp.sqrt(5-aux1), 1/3*jnp.sqrt(5+aux1)])
    weights = jnp.array([(322-aux2)/900, (322+aux2)/900, 128/225, (322+aux2)/900, (322-aux2)/900])
    x1, x2 = coords
    h = x2 - x1

    def branch_with_split(_):
        xmid = 0.5
        transformed_nodes_1 = 0.5 * (xmid - x1) * nodes + 0.5 * (x1 + xmid)
        transformed_weights_1 = 0.5 * (xmid - x1) * weights
        transformed_nodes_2 = 0.5 * (x2 - xmid) * nodes + 0.5 * (xmid + x2)
        transformed_weights_2 = 0.5 * (x2 - xmid) * weights
        sigma_int_1 = jnp.sum(transformed_weights_1 * jnp.array([SIGMA(x) for x in transformed_nodes_1]))
        sigma_int_2 = jnp.sum(transformed_weights_2 * jnp.array([SIGMA(x) for x in transformed_nodes_2]))
        return jnp.array([[sigma_int_1, -sigma_int_1], [-sigma_int_1, sigma_int_1]]) / (h**2) + \
               jnp.array([[sigma_int_2, -sigma_int_2], [-sigma_int_2, sigma_int_2]]) / (h**2)

    def branch_without_split(_):
        transformed_nodes = 0.5 * (x2 - x1) * nodes + 0.5 * (x1 + x2)
        transformed_weights = 0.5 * (x2 - x1) * weights
        sigma_int = jnp.sum(transformed_weights * jnp.array([SIGMA(x) for x in transformed_nodes]))
        return jnp.array([[sigma_int, -sigma_int], [-sigma_int, sigma_int]]) / (h**2)

    result = lax.cond(
        (x1 < 0.5) & (x2 > 0.5),
        branch_with_split,
        branch_without_split,
        operand=None  # No additional argument is needed
    )

    return result


def element_load(coords):
    aux1 = 2*jnp.sqrt(10/7)
    aux2 = 13*jnp.sqrt(70)
    nodes = jnp.array([-1/3*jnp.sqrt(5+aux1), -1/3*jnp.sqrt(5-aux1), 0, 1/3*jnp.sqrt(5-aux1), 1/3*jnp.sqrt(5+aux1)])
    weights = jnp.array([(322-aux2)/900, (322+aux2)/900, 128/225, (322+aux2)/900, (322-aux2)/900])
    x1, x2 = coords

    transformed_nodes = 0.5 * (x2 - x1) * nodes + 0.5 * (x1 + x2)
    transformed_weights = 0.5 * (x2 - x1) * weights

    problem_test = problem(problem_number)
    f = problem_test.f
    phi = lambda x: jnp.array([(x2-x)/(x2-x1), (x-x1)/(x2-x1)])
    # Evaluate the integral using the transformed nodes and weights
    return sum(w * f(x) * phi(x) for x, w in zip(transformed_nodes, transformed_weights))


# Assemble global stiffness matrix and load vector
def assemble(n_elements, node_coords, element_length, n_nodes, SIGMA):
    element_nodes = jnp.stack((jnp.arange(0, n_elements), jnp.arange(1, n_elements+1)), axis=1) 
    coords = node_coords[element_nodes]
    h_values = element_length

    def element_stiffness_with_sigma(coord):
        return element_stiffness(coord, SIGMA)

    ke_values = jax.vmap(element_stiffness_with_sigma)(coords)
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
    F = F - K[:, -1] * bc_g1

    K = K.at[0, :].set(0)
    K = K.at[:, 0].set(0)
    K = K.at[-1, :].set(0)
    K = K.at[:, -1].set(0)
    K = K.at[0, 0].set(1)
    K = K.at[-1, -1].set(1)

    F = F.at[0].set(bc_g0)
    F = F.at[-1].set(bc_g1)
    # F = F.at[-1].set(F[-1]+0.7)

    return K, F

# Solve the system
def solve(theta, sigma):
    n_nodes = theta.shape[1] + 1
    n_elements = n_nodes - 1
    node_coords = softmax_nodes(theta)
    element_length = node_coords[1:] - node_coords[:-1]

    def sigma_fn(x):
        return sigma*x + 0.1
    
    K, F = assemble(n_elements, node_coords, element_length, n_nodes, sigma_fn)
    K, F = apply_boundary_conditions(K, F)

    u = jnp.linalg.solve(K, F)

    return node_coords, u

# Solve the system
def solve_and_loss(theta, sigma):
    problem_test = problem(problem_number)
    bc_g0 = problem_test.g0
    bc_g1 = problem_test.g1

    def sigma_fn(x):
        return sigma*x + 0.1
    n_nodes = theta.shape[1] + 1
    n_elements = n_nodes - 1
    node_coords = softmax_nodes(theta)
    element_length = node_coords[1:] - node_coords[:-1]

    K, F = assemble(n_elements, node_coords, element_length, n_nodes, sigma_fn)
    K, F = apply_boundary_conditions(K, F)
    u = jnp.linalg.solve(K, F) - (bc_g0*(1-node_coords) + bc_g1*(node_coords-0))
    # u = jnp.linalg.solve(K, F)
    
    loss = (0.5*jnp.dot(u, jnp.dot(K, u)) - jnp.dot(F, u))

    return loss 


# # Define the problem domain and mesh
# n_elements = 10  # Number of elements
# n_nodes = n_elements + 1

# # Run the solver
# theta = jax.random.uniform(key=jax.random.PRNGKey(10),shape=(1,n_nodes))

# # node_coords, u = solve(theta)
# node_coords, u, val = solve_and_loss(theta)

# # # Output results
# # print("Node coordinates:", node_coords)
# # print("Solution u:", u)
# print(val)

# import matplotlib.pyplot as plt
# from matplotlib import rcParams


# # rcParams['font.family'] = 'serif'
# # rcParams['font.size'] = 18
# # rcParams['legend.fontsize'] = 17
# # rcParams['mathtext.fontset'] = 'cm'
# # rcParams['axes.labelsize'] = 19


# # # Generate a list of x values for visualization
# # xlist = node_coords

# # ## ---------
# # # SOLUTION
# ## ---------

# fig, ax = plt.subplots()
# # Plot the approximate solution obtained from the trained model
# plt.plot(node_coords, u, color='b')

# plt.legend(['u_approx', 'u_exact'])

# ax.grid(which = 'both', axis = 'both', linestyle = ':', color = 'gray')
# plt.tight_layout()

# plt.savefig('plot.png')
# plt.show()

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
        },
        {
            "f": lambda x: 4*jnp.pi**2*jnp.sin(2*jnp.pi*x),  # Singular source term
            "g0": 0,  # Dirichlet boundary condition at x=0
            "g1": 0,  # Dirichlet boundary condition at x=1
            "sigma": lambda x: jnp.piecewise(x, [x < 0.5, x >= 0.5], [1, 10]), # 1 + 9/(1+jnp.exp(-(x-0.5)*1000)) # Piecewise coefficient 
        }
    ]

    # Ensure problem_number is within bounds
    if problem_number < 0 or problem_number >= len(problems_data):
        raise ValueError(f"Invalid problem number: {problem_number}. Must be between 0 and {len(problems_data) - 1}.")

    # Retrieve problem data and unpack into the Elliptic1D class
    data = problems_data[problem_number]
    return Elliptic1D(**data)