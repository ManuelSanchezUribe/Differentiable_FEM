import jax
import jax.numpy as jnp
from jax import jit
import keras
from functools import partial

global problem_number
problem_number=0

#@jit
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
    coords = jnp.zeros((nx*ny,2), dtype=jnp.float64)
    elements = jnp.zeros((n_el,4), dtype=jnp.int32)

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
    def element_stiffness_fn(length):
        return element_stiffness(length)
    
    # Compute stiffness matrices for all elements
    element_stiffness_matrices = jax.vmap(element_stiffness_fn)(element_length)
    
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
def apply_boundary_conditions(K, F, dirichlet_nodes):
    problem_test = problem(problem_number)

    # F = F - K[:, 0] * bc_g0
    # F = F - K[:, -1] * bc_g1

    K = K.at[dirichlet_nodes, :].set(0)
    K = K.at[:, dirichlet_nodes].set(0)
    # K = K.at[-1, :].set(0)
    # K = K.at[:, -1].set(0)
    K = K.at[dirichlet_nodes, dirichlet_nodes].set(1)
    # K = K.at[-1, -1].set(1)

    F = F.at[dirichlet_nodes].set(0)
    # F = F.at[-1].set(bc_g1)
    # F = F.at[-1].set(F[-1]+0.7)

    return K

# Solve the system
def solve(theta):
    nx = int(theta.shape[0]/2)
    ny = nx
    
    node_coords_x, node_coords_y  = softmax_nodes(theta)
    # node_coords_x = jnp.linspace(0, 1, nx)
    # node_coords_y = jnp.linspace(0, 1, ny)
    coords, elements = generate_mesh(nx, ny, node_coords_x, node_coords_y)
    n_elements = elements.shape[0]
    n_nodes = coords.shape[0]

    dirichlet_nodes = jnp.append(jnp.arange(nx),nx*jnp.arange(1,ny))
    neumann_nodes = jnp.append(nx*jnp.arange(1,ny)-1, jnp.arange((ny-1)*nx, ny*nx))

    dirichlet_nodes = jnp.append(dirichlet_nodes, neumann_nodes)
    dirichlet_nodes = jnp.unique(dirichlet_nodes)

    element_length = jnp.zeros((n_elements, 2))

    for e in range(n_elements):
        x1, y1 = coords[elements[e, 0], :]
        x2, y2 = coords[elements[e, 2], :]
        element_length = element_length.at[e,:].set([x2 - x1, y2 - y1])

    K = assemble_stiffness(n_elements, elements, element_length, n_nodes)
    F = load_vector(coords, elements)

    K = apply_boundary_conditions(K, F, dirichlet_nodes)
    u = jnp.linalg.solve(K, F)

    return coords, u

# Solve the system
def solve_and_loss(theta):
    nx = int(theta.shape[0]/2)
    ny = nx
    node_coords_x, node_coords_y  = softmax_nodes(theta)
    # node_coords_x = jnp.linspace(0, 1, nx)
    # node_coords_y = jnp.linspace(0, 1, ny)
    coords, elements = generate_mesh(nx, ny, node_coords_x, node_coords_y)
    dirichlet_nodes = jnp.append(jnp.arange(nx),nx*jnp.arange(1,ny))
    neumann_nodes = jnp.append(nx*jnp.arange(1,ny)-1, jnp.arange((ny-1)*nx, ny*nx))

    dirichlet_nodes = jnp.append(dirichlet_nodes, neumann_nodes)
    dirichlet_nodes = jnp.unique(dirichlet_nodes)

    n_elements = elements.shape[0]
    n_nodes = coords.shape[0]

    element_length = jnp.zeros((n_elements, 2))

    for e in range(n_elements):
        x1, y1 = coords[elements[e, 0], :]
        x2, y2 = coords[elements[e, 2], :]
        element_length = element_length.at[e,:].set([x2 - x1, y2 - y1])

    K = assemble_stiffness(n_elements, elements, element_length, n_nodes)
    F = load_vector(coords, elements)

    K = apply_boundary_conditions(K, F, dirichlet_nodes)
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

# # Graficar el resultado
# plt.figure(figsize=(8, 6))
# plt.tricontourf(triangulation, u, cmap='viridis')
# plt.colorbar(label='u (Solución)')
# plt.title('Resultados de elementos finitos en 2D')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.axis('equal')  # Mantener proporciones
# plt.show()