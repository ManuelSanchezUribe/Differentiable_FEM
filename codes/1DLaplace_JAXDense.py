import jax
import jax.numpy as jnp
from jax import jit

# Define the problem domain and mesh
n_elements = 10  # Number of elements
n_nodes = n_elements + 1
length = 1.0

# Define a nonequidistant mesh (e.g., using a quadratic distribution)
node_coords = jnp.sort(jnp.linspace(0, 1, n_nodes)**2)
element_length = node_coords[1:] - node_coords[:-1]

# Define source function f(x)
def f(x):
    return 0

# Boundary conditions
def g0():
    return 0.5  # Value of u at x = 0
def g1():
    return -0.5  # Value of u at x = 1

# Element stiffness matrix and load vector
def element_stiffness(h):
    return jnp.array([[1, -1], [-1, 1]]) / h

def element_load(coords):
    x1, x2 = coords
    midpoint = (x1 + x2) / 2
    h = x2 - x1
    return h * jnp.array([f(midpoint), f(midpoint)]) / 2

# Assemble global stiffness matrix and load vector
def assemble():
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
    F = F - K[:, -1] * bc_g1

    K = K.at[0, :].set(0)
    K = K.at[:, 0].set(0)
    K = K.at[-1, :].set(0)
    K = K.at[:, -1].set(0)
    K = K.at[0, 0].set(1)
    K = K.at[-1, -1].set(1)

    F = F.at[0].set(bc_g0)
    F = F.at[-1].set(bc_g1)

    return K, F

# Solve the system
@jit
def solve():
    K, F = assemble()
    K, F = apply_boundary_conditions(K, F)
    u = jnp.linalg.solve(K, F)
    return u

# Run the solver
u = solve()

# Output results
print("Node coordinates:", node_coords)
print("Solution u:", u)

import matplotlib.pyplot as plt
from matplotlib import rcParams


rcParams['font.family'] = 'serif'
rcParams['font.size'] = 18
rcParams['legend.fontsize'] = 17
rcParams['mathtext.fontset'] = 'cm'
rcParams['axes.labelsize'] = 19


# Generate a list of x values for visualization
xlist = node_coords

## ---------
# SOLUTION
## ---------

fig, ax = plt.subplots()
# Plot the approximate solution obtained from the trained model
plt.plot(xlist, u, color='b')

plt.legend(['u_approx', 'u_exact'])

ax.grid(which = 'both', axis = 'both', linestyle = ':', color = 'gray')
plt.tight_layout()
plt.show()