import jax
import jax.numpy as jnp
from jax import jit
import keras


def softmax_nodes(params):
    # Compute the softmax values
    softmax_values = jax.nn.softmax(params)
    # softmax_values = jax.nn.softmax(params)

    # print(paa)
    # Compute the cumulative sum of the softmax values
    cumulative_sum = jnp.cumsum(softmax_values)

    return cumulative_sum

# Define source function f(x)
def f(x):
    # return 0.7*0.3*x**(-1.3) + 1.7*0.7*x**(-0.3)
    # return 0
    return 0.7*0.3*x**(-1.3)

# Boundary conditions
def g0():
    return 0  # Value of u at x = 0
    # return 0.5
def g1():
    return 0  # Value of u at x = 1
    # return -0.5

# Element stiffness matrix and load vector
def element_stiffness(h):
    return jnp.array([[1, -1], [-1, 1]]) / h

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
    return h * jnp.array([f(pt1)*phiatpt1 + f(pt2)*phiatpt2, f(pt1)*phiatpt2 + f(pt2)*phiatpt1]) / 2

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
    bc_g0 = g0()
    bc_g1 = g1()

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

def eval_loss(K,F,u):
    return 0.5*jnp.dot(u, jnp.dot(K, u)) + jnp.dot(F, u)

# Solve the system
def solve(theta):
    n_nodes = theta.shape[1]
    n_elements = n_nodes - 1
    node_coords = softmax_nodes(theta)
    element_length = node_coords[1:] - node_coords[:-1]

    K, F = assemble(n_elements, node_coords, element_length, n_nodes)
    K, F = apply_boundary_conditions(K, F)
    u = jnp.linalg.solve(K, F)

    return node_coords, u

# Solve the system
def solve_and_loss(theta):
    n_nodes = theta.shape[1]
    n_elements = n_nodes - 1
    node_coords = softmax_nodes(theta)
    element_length = node_coords[1:] - node_coords[:-1]

    K, F = assemble(n_elements, node_coords, element_length, n_nodes)
    K, F = apply_boundary_conditions(K, F)
    u = jnp.linalg.solve(K, F)
    
    loss = 0.5*jnp.dot(u, jnp.dot(K, u)) - jnp.dot(F, u)

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