import jax
import jax.numpy as jnp
from jax import jit
import jax.experimental.sparse as sparse
from functools import partial

global problem_number
problem_number=2

@jit
def softmax_nodes(params):
    n_nodes = params.shape[1]
    # Compute the softmax values
    softmax_values = jax.nn.softmax(params)

    # Compute the cumulative sum of the softmax values in X axis
    cumulative_sum_x = jnp.cumsum(softmax_values[0, 0:int(n_nodes/2)])
    cumulative_sum_x_with_zero = jnp.insert(cumulative_sum_x, 0, 0)

    # Compute the cumulative sum of the softmax values in Y axis
    cumulative_sum_y = jnp.cumsum(softmax_values[0, int(n_nodes/2):])
    cumulative_sum_y_with_zero = jnp.insert(cumulative_sum_y, 0, 0)

    return cumulative_sum_x_with_zero, cumulative_sum_y_with_zero

@jit
def element_stiffness(h):
    return jnp.array([[1, -1], [-1, 1]]) / h

@jit
def element_load(coords):
    x1, x2   = coords
    p1       = -1/jnp.sqrt(3)
    p2       = 1/jnp.sqrt(3)
    pt1      = (x2 - x1) * p1 / 2 + (x2 + x1) / 2
    pt2      = (x2 - x1) * p2 / 2 + (x2 + x1) / 2
    phiatpt1 = (p2+1)/2
    phiatpt2 = (1+p1)/2
    h        = x2 - x1
    problem_test = problem(problem_number)
    f            = problem_test.f
    return h * jnp.array([f(pt1)*phiatpt1 + f(pt2)*phiatpt2, f(pt1)*phiatpt2 + f(pt2)*phiatpt1]) / 2

# Assemble global stiffness matrix and load vector
@partial(jax.jit, static_argnames=['n_elements', 'n_nodes'])
def assemble(n_elements, node_coords, element_length, n_nodes):
    element_nodes = jnp.stack((jnp.arange(0, n_elements), jnp.arange(1, n_elements+1)), axis=1) 
    coords        = node_coords[element_nodes]
    h_values      = element_length
    lenval        = 3*(n_nodes-2) + 4
    values        = jnp.zeros((lenval))

    fe_values = jax.vmap(element_load)(coords)
    ke_values = jax.vmap(element_stiffness)(h_values)

    # Compute the indices for vectorized updates
    indices = jnp.arange(n_elements)  # Array of element indices
    flat_indices = (3 * indices[:, None] + jnp.arange(4)).reshape(-1)  # Compute flattened indices

    # Flatten ke_values to match flat_indices
    flattened_ke_values = ke_values.reshape(-1)

    # Perform the update in a vectorized manner
    values = values.at[flat_indices].add(flattened_ke_values)
    values = values.at[1:3].set(0)
    values = values.at[0].set(1)

    rows = 3 * jnp.arange(n_nodes+1) - 1
    rows = rows.at[0].set(0)
    rows = rows.at[-1].add(-1)

    start_indices = jnp.maximum(0, jnp.arange(n_nodes - 1) - 1)
    end_indices = jnp.minimum(jnp.arange(n_nodes - 1) + 2, n_nodes)

    # Compute ranges for all elements at once
    max_range = 3  # Maximum possible range size (i.e., 3 elements: i-1, i, i+1)
    range_matrix = jnp.arange(max_range) + start_indices[:, None]  # Broadcast addition
    valid_mask = (range_matrix < end_indices[:, None]) & (range_matrix >= start_indices[:, None])

    # Mask out invalid elements and flatten
    valid_values = jnp.where(valid_mask, range_matrix, -1)

    # Append the final elements
    cols = jnp.array([0,1])
    cols = jnp.append(cols, valid_values[1:])
    cols = jnp.append(cols, jnp.array([n_nodes - 2, n_nodes - 1]))
    K    = jax.experimental.sparse.CSR((values, cols, rows), shape=(n_nodes, n_nodes))

    F = jnp.zeros(n_nodes)

    F = F.at[element_nodes[:, 0]].add(fe_values[:, 0])
    F = F.at[element_nodes[:, 1]].add(fe_values[:, 1])

    return K, F

# Apply boundary conditions
@jit
def apply_boundary_conditions(K, F):
    problem_test = problem(problem_number)
    bc_g0 = problem_test.g0
    bc_g1 = problem_test.g1
    # bc_g0 = g0()
    # bc_g1 = g1()

    # F = F - K[:, 0] * bc_g0
    # F = F - K[:, -1] * bc_g1

    # K = K.at[0, :].set(0)
    # K = K.at[:, 0].set(0)
    # K = K.at[-1, :].set(0)
    # K = K.at[:, -1].set(0)
    # K = K.at[0, 0].set(1)
    # K = K.at[-1, -1].set(1)

    F = F.at[0].set(bc_g0)
    # F = F.at[-1].set(bc_g1)
    F = F.at[-1].set(F[-1]+0.7)

    return K, F

# Solve the system

def solve(theta):
    n_nodes = theta.shape[1] + 1
    n_elements = n_nodes - 1
    node_coords = softmax_nodes(theta)
    element_length = node_coords[1:] - node_coords[:-1]

    K, F = assemble(n_elements, node_coords, element_length, n_nodes)
    K, F = apply_boundary_conditions(K, F)
    u  = sparse.linalg.spsolve(K.data, K.indices, K.indptr, F, reorder = 0)

    return node_coords, u

# Solve the system
@jit
def solve_and_loss(theta):
    problem_test = problem(problem_number)
    bc_g0 = problem_test.g0
    bc_g1 = problem_test.g1

    n_nodes        = theta.shape[1] + 1
    n_elements     = n_nodes - 1
    node_coords    = softmax_nodes(theta)
    element_length = node_coords[1:] - node_coords[:-1]

    K, F = assemble(n_elements, node_coords, element_length, n_nodes)
    K, F = apply_boundary_conditions(K, F)
    
    u  = sparse.linalg.spsolve(K.data, K.indices, K.indptr, F, reorder = 0)
    loss = 0.5 * u @ (K @ u) - F @ u
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
        self.a     = 0.0  # Left endpoint of the domain
        self.b     = 1.0  # Right endpoint of the domain
        self.f     = f
        self.g0    = g0
        self.g1    = g1
        self.sigma = sigma
        self.u     = u  # Analytical solution, if provided

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
    ]

    # Ensure problem_number is within bounds
    if problem_number < 0 or problem_number >= len(problems_data):
        raise ValueError(f"Invalid problem number: {problem_number}. Must be between 0 and {len(problems_data) - 1}.")

    # Retrieve problem data and unpack into the Elliptic1D class
    data = problems_data[problem_number]
    return Elliptic1D(**data)