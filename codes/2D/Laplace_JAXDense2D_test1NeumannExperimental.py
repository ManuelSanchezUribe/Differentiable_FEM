import jax
import jax.numpy as jnp
from jax import jit
import keras
from functools import partial
from jax import config; config.update("jax_enable_x64", True)


global problem_number
problem_number=4

#@jit
def softmax_nodes(params):
    n_nodes = params.shape[1]
    # Compute the softmax values
    # softmax_values_x = jax.nn.softmax(params[0, 0:int(n_nodes/2)])
    # softmax_values_y = jax.nn.softmax(params[0, int(n_nodes/2):])
    softmax_values_x = jax.nn.softmax(params[0, 0:int(n_nodes -2)])
    softmax_values_y = jax.nn.softmax(params[0, int(n_nodes -2):])

    # Compute the cumulative sum of the softmax values in X axis
    cumulative_sum_x = jnp.cumsum(softmax_values_x)
    cumulative_sum_x_with_zero = jnp.insert(cumulative_sum_x, 0, 0)

    # Compute the cumulative sum of the softmax values in Y axis
    cumulative_sum_y = jnp.cumsum(softmax_values_y)
    cumulative_sum_y_with_zero = jnp.insert(cumulative_sum_y, 0, 0)

    return cumulative_sum_x_with_zero, cumulative_sum_y_with_zero


values_phi0_ = jnp.array([[0.9083804012656871,  0.7331497981296533,  0.47654496148466596, 0.21994012483967862, 0.04470952170364481],
                         [0.7331497981296533,  0.591721954534264, 0.38461732752642075, 0.17751270051857745,  0.036084856923188136],
                         [0.47654496148466596, 0.38461732752642075, 0.25,       0.11538267247357925, 0.02345503851533401],
                         [0.21994012483967862, 0.17751270051857745,  0.11538267247357925, 0.053252644428581054, 0.010825220107479883],
                         [0.04470952170364481, 0.036084856923188136, 0.02345503851533401, 0.010825220107479883, 0.002200555327023207]])
# jnp.set_printoptions(precision=None, threshold =10000000000)

# Generate a structured grid
def generate_mesh(nx, ny, x, y):
    # x = jnp.linspace(x_min, x_max, nx)
    # y = jnp.linspace(y_min, y_max, ny)
    n_el = (nx-1)*(ny-1)
    coords = jnp.zeros((nx*ny,2))
    elements = jnp.zeros((n_el,4), dtype=jnp.int64)

    for i in range(nx):
        for j in range(ny):
            ind = j*nx + i
            coords = coords.at[ind,0].set(x[i])
            coords = coords.at[ind,1].set(y[j])
    for i in range(nx-1):
        for j in range(ny-1):
            ind = j*(nx-1) + i
            zero = ind + j
            one = zero + 1
            three = ind + (nx-1) + j + 1
            two = three + 1
            elements = elements.at[ind,:].set([zero, one, two, three])

    return coords, elements

# Assemble the stiffness matrix
def element_stiffness(hvalues):
    hx, hy = hvalues
    array_x = jnp.array([[2, -2, -1, 1],
                        [-2, 2, 1, -1],
                        [-1, 1, 2, -2],
                        [1, -1, -2, 2]]) * hy / 6 / hx
    array_y = jnp.array([[2, 1, -1, -2],
                        [1, 2, -2, -1],
                        [-1, -2, 2, 1],
                        [-2, -1, 1, 2]]) * hx / 6 / hy
    
    return array_x + array_y 


def assemble_stiffness(n_elements, elements, element_length, n_nodes):
    # Function to compute stiffness matrix for a single element
    
    # Compute stiffness matrices for all elements
    element_stiffness_matrices = jax.vmap(element_stiffness)(element_length)
    
    # Generate row and column indices for scatter addition
    row_indices = elements[:, None, :].repeat(elements.shape[1], axis=1)
    col_indices = elements[:, :, None].repeat(elements.shape[1], axis=2)
    
    # Use scatter addition to accumulate values into the global matrix
    K = jnp.zeros((n_nodes, n_nodes))
    K = K.at[row_indices, col_indices].add(element_stiffness_matrices)

    return K


# Assemble the load vector
def load_vector(coords, elements):
    problem_test = problem(problem_number)
    f = problem_test.f
    aux1 = 2*jnp.sqrt(10/7)
    aux2 = 13*jnp.sqrt(70)
    nodes = jnp.array([-1/3*jnp.sqrt(5+aux1), -1/3*jnp.sqrt(5-aux1), 0, 1/3*jnp.sqrt(5-aux1), 1/3*jnp.sqrt(5+aux1)])
    weights = jnp.array([(322-aux2)/900, (322+aux2)/900, 128/225, (322+aux2)/900, (322-aux2)/900])

    def compute_element(e):
        x1, y1 = coords[elements[e, 0], :]
        x2, y2 = coords[elements[e, 2], :]
        hx, hy = x2 - x1, y2 - y1
        transf_nodes_x = 0.5 * hx * nodes + 0.5 * (x1 + x2)
        transf_nodes_y = 0.5 * hy * nodes + 0.5 * (y1 + y2)

        # Compute sums using broadcasting
        tx, ty = jnp.meshgrid(transf_nodes_x, transf_nodes_y, indexing="ij")
        wx, wy = jnp.meshgrid(weights, weights, indexing="ij")
        jacobian = hx * hy * 0.25

        f_vals = f(tx, ty)
        sum0 = jnp.sum(f_vals * values_phi0_ * wx * wy * jacobian)
        sum1 = jnp.sum(f_vals * values_phi0_[:, ::-1] * wx * wy * jacobian)
        sum2 = jnp.sum(f_vals * values_phi0_[::-1, ::-1] * wx * wy * jacobian)
        sum3 = jnp.sum(f_vals * values_phi0_[::-1, :] * wx * wy * jacobian)

        return elements[e, :], jnp.array([sum0, sum1, sum2, sum3])

    F = jnp.zeros((coords.shape[0]))
    n_el = elements.shape[0]
    element_indices, element_contributions = jax.vmap(compute_element)(jnp.arange(n_el))
    F = F.at[element_indices].add(element_contributions)

    return F


# Apply boundary conditions
def apply_boundary_conditions(K, F, dirichlet_nodes, neumann_edges, elements, coords):
    problem_test = problem(problem_number)
    K = K.at[dirichlet_nodes, :].set(0)
    K = K.at[:, dirichlet_nodes].set(0)
    K = K.at[dirichlet_nodes, dirichlet_nodes].set(1)

    F = F.at[dirichlet_nodes].set(0)

    aux1 = 2*jnp.sqrt(10/7)
    aux2 = 13*jnp.sqrt(70)
    nodes = jnp.array([-1/3*jnp.sqrt(5+aux1), -1/3*jnp.sqrt(5-aux1), 0, 1/3*jnp.sqrt(5-aux1), 1/3*jnp.sqrt(5+aux1)])
    weights = jnp.array([(322-aux2)/900, (322+aux2)/900, 128/225, (322+aux2)/900, (322-aux2)/900])
    
    # for i in range(neumann_edges.shape[0]//2):
    # for i in range(neumann_edges.shape[0]//3):
    for i in range(2):
        g = problem_test.gr
        e = neumann_edges[i, 0]
        n1 = neumann_edges[i, 1]
        n2 = neumann_edges[i, 2]
        e = e.astype(jnp.int64)
        n1 = n1.astype(jnp.int64)
        n2 = n2.astype(jnp.int64)
        x1, y1 = coords[elements[e, n1], :]
        x2, y2 = coords[elements[e, n2], :]
        norm = 0.5 * jnp.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        transformed_x = x1 + 0.5 * (nodes + 1) * (x2 - x1)
        transformed_y = y1 + 0.5 * (nodes + 1) * (y2 - y1)
        sum0 = jnp.sum(norm * weights * g(transformed_x, transformed_y) * (0.5 * (1 - nodes)))
        sum1 = jnp.sum(norm * weights * g(transformed_x, transformed_y) * (0.5 * (1 + nodes)))

        F = F.at[elements[e, n1]].add(sum0)
        F = F.at[elements[e, n2]].add(sum1)

    # for i in range(neumann_edges.shape[0]//2, neumann_edges.shape[0]):
    # for i in range(neumann_edges.shape[0]//3, neumann_edges.shape[0]):
    for i in range(i, neumann_edges.shape[0]):
        g = problem_test.gu
        e = neumann_edges[i, 0]
        n1 = neumann_edges[i, 1]
        n2 = neumann_edges[i, 2]
        e = e.astype(jnp.int64)
        n1 = n1.astype(jnp.int64)
        n2 = n2.astype(jnp.int64)
        x1, y1 = coords[elements[e, n1], :]
        x2, y2 = coords[elements[e, n2], :]
        norm = 0.5 * jnp.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        transformed_x = x1 + 0.5 * (nodes + 1) * (x2 - x1)
        transformed_y = y1 + 0.5 * (nodes + 1) * (y2 - y1)
        sum0 = jnp.sum(norm * weights * g(transformed_x, transformed_y) * (0.5 * (1 - nodes)))
        sum1 = jnp.sum(norm * weights * g(transformed_x, transformed_y) * (0.5 * (1 + nodes)))

        F = F.at[elements[e, n1]].add(sum0)
        F = F.at[elements[e, n2]].add(sum1)

    return K, F

# Solve the system
def solve(theta):
    # nx = int(theta.shape[1]/2) + 1
    # ny = nx

    nx = int(theta.shape[1] - 2) + 1
    ny = 2
    
    node_coords_x, node_coords_y  = softmax_nodes(theta)
    # node_coords_x = jnp.linspace(0, 1, nx)
    # node_coords_y = jnp.linspace(0, 1, ny)
    coords, elements = generate_mesh(nx, ny, node_coords_x, node_coords_y)
    n_elements = elements.shape[0]
    n_nodes = coords.shape[0]

    # dirichlet_nodes = jnp.append(jnp.arange(nx),nx*jnp.arange(1,ny))
    dirichlet_nodes = nx*jnp.arange(1,ny)
    
    ind1 = jnp.arange(ny-1, dtype=jnp.int64) * (nx-1) + (nx-2)
    # ind2 = (ny-2) * (nx-1) + jnp.arange(nx-1, dtype=jnp.int64)
    ind2 = jnp.append(jnp.arange(nx-1), (ny-2) * (nx-1) + jnp.arange(nx-1, dtype=jnp.int64))

    # local1 = jnp.append(jnp.ones(ny-1), 2 * jnp.ones(nx-1))
    local1 = jnp.append(jnp.ones(ny-1), jnp.zeros(nx-1))
    local1 = jnp.append(local1, 2 * jnp.ones(nx-1))
    # local2 = jnp.append(2 * jnp.ones(ny-1), 3 * jnp.ones(nx-1))
    local2 = jnp.append(2 * jnp.ones(ny-1),jnp.ones(nx-1))
    local2 = jnp.append(local2, 3 * jnp.ones(nx-1))

    side = jnp.append(jnp.zeros(ny-1), jnp.ones(2*(nx-1)))
    ind = jnp.append(ind1, ind2)

    neumann_edges = jnp.reshape(jnp.concatenate([ind, local1, local2, side], axis=0), (2*(nx-1) + (ny-1), 4), order='F')

    # Extract the coordinates for the start and end points of each element
    start_coords = coords[elements[:, 0], :]
    end_coords = coords[elements[:, 2], :]

    # Compute element lengths in a vectorized manner
    element_length = end_coords - start_coords

    K = assemble_stiffness(n_elements, elements, element_length, n_nodes)
    F = load_vector(coords, elements)

    K, F = apply_boundary_conditions(K, F, dirichlet_nodes, neumann_edges, elements, coords)
    u = jnp.linalg.solve(K, F)

    return coords, u

# Solve the system
def solve_and_loss(theta):
    # nx = int(theta.shape[1]/2) + 1
    # ny = nx

    nx = int(theta.shape[1] - 2) + 1
    ny = 2
    
    node_coords_x, node_coords_y  = softmax_nodes(theta)
    # node_coords_x = jnp.linspace(0, 1, nx)
    # node_coords_y = jnp.linspace(0, 1, ny)
    coords, elements = generate_mesh(nx, ny, node_coords_x, node_coords_y)
    n_elements = elements.shape[0]
    n_nodes = coords.shape[0]

    # dirichlet_nodes = jnp.append(jnp.arange(nx),nx*jnp.arange(1,ny))
    dirichlet_nodes = nx*jnp.arange(1,ny)
    
    ind1 = jnp.arange(ny-1, dtype=jnp.int64) * (nx-1) + (nx-2)
    # ind2 = (ny-2) * (nx-1) + jnp.arange(nx-1, dtype=jnp.int64)
    ind2 = jnp.append(jnp.arange(nx-1), (ny-2) * (nx-1) + jnp.arange(nx-1, dtype=jnp.int64))

    # local1 = jnp.append(jnp.ones(ny-1), 2 * jnp.ones(nx-1))
    local1 = jnp.append(jnp.ones(ny-1), jnp.zeros(nx-1))
    local1 = jnp.append(local1, 2 * jnp.ones(nx-1))
    # local2 = jnp.append(2 * jnp.ones(ny-1), 3 * jnp.ones(nx-1))
    local2 = jnp.append(2 * jnp.ones(ny-1),jnp.ones(nx-1))
    local2 = jnp.append(local2, 3 * jnp.ones(nx-1))

    side = jnp.append(jnp.zeros(ny-1), jnp.ones(2*(nx-1)))
    ind = jnp.append(ind1, ind2)

    neumann_edges = jnp.reshape(jnp.concatenate([ind, local1, local2, side], axis=0), (2*(nx-1) + (ny-1), 4), order='F')

    # Extract the coordinates for the start and end points of each element
    start_coords = coords[elements[:, 0], :]
    end_coords = coords[elements[:, 2], :]

    # Compute element lengths in a vectorized manner
    element_length = end_coords - start_coords

    K = assemble_stiffness(n_elements, elements, element_length, n_nodes)
    F = load_vector(coords, elements)

    K, F = apply_boundary_conditions(K, F, dirichlet_nodes, neumann_edges, elements, coords)
    u = jnp.linalg.solve(K, F)

    loss = 0.5*jnp.dot(u, jnp.dot(K, u)) - jnp.dot(F, u)

    return loss

class Elliptic1D:
    def __init__(self, f, gu, gr, sigma, ritz_value, u=None):
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
        self.gu = gu
        self.gr = gr
        self.sigma = sigma
        self.ritz_value = ritz_value
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
            "f": lambda x, y: 2*jnp.pi**2*jnp.sin(jnp.pi*x)*jnp.sin(jnp.pi*y),  # Zero source term
            "gu": lambda x, y:jnp.pi*jnp.cos(jnp.pi*y)*jnp.sin(jnp.pi*x),  # Neumman Boundary condition for y = 1
            "gr": lambda x, y:jnp.pi*jnp.cos(jnp.pi*x)*jnp.sin(jnp.pi*y),  # Neumman Boundary condition for x = 1
            "sigma": lambda x, y: 1,  # Constant coefficient sigma(x),
            "ritz_value": -0.25*jnp.pi**2, # Ritz value for the problem
        },
        {
            "f": lambda x, y: -(jnp.atan(10*y - 0.5) + jnp.atan(0.5))*200*(0.5 - 10*x)/(1 + (10*x - 0.5)**2)**2 \
                - (jnp.atan(10*x - 0.5) + jnp.atan(0.5))*200*(0.5 - 10*y)/(1 + (10*y - 0.5)**2)**2,  # Sinusoidal source term
            "gu": lambda x, y: (jnp.atan(10*x - 0.5) + jnp.atan(0.5))*10/(1 + (10*y - 0.5)**2),  # Neumman Boundary condition for y = 1  
            "gr": lambda x, y: (jnp.atan(10*y - 0.5) + jnp.atan(0.5))*10/(1 + (10*x - 0.5)**2),  # Neumman Boundary condition for x = 1
            "sigma": lambda x, y: 1,  # Constant coefficient sigma(x),
            "ritz_value": -34.3027, # Ritz value for the problem
        },
        {
            "f": lambda x, y: 0.21*x**(-1.3)*y, # Singular source term
            "gu": lambda x, y: x**(0.7),  # Neumman Boundary condition for y = 1 
            "gr": lambda x, y: 0.7*y,  # Neumman Boundary condition for x = 1
            "sigma": lambda x, y: 1,  # Constant coefficient sigma(x),
            "ritz_value": -34.3027, # Ritz value for the problem
        },
        {
            "f": lambda x, y: -2*y**2 - 2*x**2, # Singular source term
            "gu": lambda x, y: 2*x**2,  # Neumman Boundary condition for y = 1 
            "gr": lambda x, y: 2*y**2,  # Neumman Boundary condition for x = 1
            "sigma": lambda x, y: 1,  # Constant coefficient sigma(x),
            "ritz_value": -34.3027, # Ritz value for the problem
        },
        {
            "f": lambda x, y: 0*0.21*x**(-1.3), # Singular source term
            "gu": lambda x, y: 0,  # Neumman Boundary condition for y = 1 
            "gr": lambda x, y: 1,  # Neumman Boundary condition for x = 1
            "sigma": lambda x, y: 1,  # Constant coefficient sigma(x),
            "ritz_value": -34.3027, # Ritz value for the problem
        },
    ]

    # Ensure problem_number is within bounds
    if problem_number < 0 or problem_number >= len(problems_data):
        raise ValueError(f"Invalid problem number: {problem_number}. Must be between 0 and {len(problems_data) - 1}.")

    # Retrieve problem data and unpack into the Elliptic1D class
    data = problems_data[problem_number]
    return Elliptic1D(**data)


# print(solve_and_loss(jnp.zeros((200))))
# coords, u = solve(jnp.zeros((200)))
# print(max(u))

# # # ## ---------
# # # # SOLUTION
# # ## ---------
# import matplotlib.pyplot as plt
# import matplotlib.tri as tri
# # # Crear el triángulo para el trazado
# triangulation = tri.Triangulation(coords[:, 0], coords[:, 1])
# print(jnp.max(u))
# # Graficar el resultado
# plt.figure(figsize=(8, 6))
# plt.tricontourf(triangulation, u, cmap='viridis')
# plt.colorbar(label='u (Solución)')
# plt.title('Resultados de elementos finitos en 2D')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.axis('equal')  # Mantener proporciones
# plt.savefig('femtest.png')
# plt.show()