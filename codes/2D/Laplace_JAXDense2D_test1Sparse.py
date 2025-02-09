########################################################################################
# Workshop: Coding for PDEs with Neural Networks
# Date: 2025-24-01
# Author: Danilo Aballay, Vicente Iligaray, Ignacio Tapia y Manuel Sánchez
########################################################################################

import jax
import jax.numpy as jnp
from jax import jit
import keras
from functools import partial
from jax import config; config.update("jax_enable_x64", True)

global problem_number
problem_number=0

# @jit
def softmax_nodes(params):
    n_nodes = params.shape[1]
    # Compute the softmax values
    softmax_values_x = jax.nn.softmax(params[0, 0:int(n_nodes/2)])
    softmax_values_y = jax.nn.softmax(params[0, int(n_nodes/2):])

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
values_phi1_ = jnp.array([[0.04470952170364481, 0.036084856923188136, 0.02345503851533401, 0.010825220107479883, 0.002200555327023207],
                         [0.21994012483967862,  0.17751270051857745, 0.11538267247357925, 0.053252644428581054,  0.010825220107479883],
                         [0.47654496148466596, 0.38461732752642075, 0.25,       0.11538267247357925, 0.02345503851533401],
                         [0.7331497981296533, 0.591721954534264,  0.38461732752642075, 0.17751270051857745, 0.036084856923188136],
                         [0.9083804012656871, 0.7331497981296533, 0.47654496148466596, 0.21994012483967862, 0.04470952170364481]])
values_phi2_ = jnp.array([[0.002200555327023207, 0.010825220107479883, 0.02345503851533401, 0.036084856923188136, 0.04470952170364481],
                         [0.010825220107479883,  0.053252644428581054, 0.11538267247357925, 0.17751270051857745,  0.21994012483967862],
                         [0.02345503851533401, 0.11538267247357925, 0.25,       0.38461732752642075, 0.47654496148466596],
                         [0.036084856923188136, 0.17751270051857745,  0.38461732752642075, 0.591721954534264, 0.7331497981296533],
                         [0.04470952170364481, 0.21994012483967862, 0.47654496148466596, 0.7331497981296533, 0.9083804012656871]])
values_phi3_ = jnp.array([[0.04470952170364481, 0.21994012483967862, 0.47654496148466596, 0.7331497981296533, 0.9083804012656871],
                         [0.036084856923188136,  0.17751270051857745, 0.38461732752642075, 0.591721954534264,  0.7331497981296533],
                         [0.02345503851533401, 0.11538267247357925, 0.25,       0.38461732752642075, 0.47654496148466596],
                         [0.010825220107479883, 0.053252644428581054,  0.11538267247357925, 0.17751270051857745, 0.21994012483967862],
                         [0.002200555327023207, 0.010825220107479883, 0.02345503851533401, 0.036084856923188136, 0.04470952170364481]])

# jnp.set_printoptions(precision=None, threshold =10000000000)

# Generate a structured grid
# @partial(jax.jit, static_argnames=['nx', 'ny'])
def generate_mesh(nx, ny, x, y):
    #n_el = (nx-1)*(ny-1)
    x_coords, y_coords = jnp.meshgrid(x, y, indexing='xy')
    coords = jnp.column_stack((x_coords.flatten(), y_coords.flatten()))
    i, j = jnp.meshgrid(jnp.arange(nx-1), jnp.arange(ny-1), indexing='xy')
    ind = j * nx + i
    zero = ind
    one = zero + 1
    three = zero + nx
    two = three + 1
    elements = jnp.stack([zero.flatten(), one.flatten(), two.flatten(), three.flatten()], axis=-1)

    return coords, elements


# Assemble the stiffness matrix
# @jit
def element_stiffness(hvalues):
    # print(hvalues)
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

# @partial(jax.jit, static_argnames=['n_nodes'])
def assemble_stiffness(n_elements, elements, element_length, n_nodes):
    # Compute stiffness matrices for all elements
    element_stiffness_matrices = jax.vmap(element_stiffness)(element_length)
    # print(element_stiffness_matrices)
    
    # Generate row and column indices for scatter addition
    row_indices = elements[:, None, :].repeat(elements.shape[1], axis=1)
    col_indices = elements[:, :, None].repeat(elements.shape[1], axis=2)
    
    # Use scatter addition to accumulate values into the global matrix
    K = jnp.zeros((n_nodes, n_nodes))
    K = K.at[row_indices, col_indices].add(element_stiffness_matrices)

    return K

def assemble_stiffness_dir_corners(corner_elements, elements, n_nodes, corner_local_nodes_non_dirichlet, corner_length):
    K = jnp.zeros((n_nodes, n_nodes))
    for i in range(corner_local_nodes_non_dirichlet.shape[0]):
        local_matrix = element_stiffness(corner_length[i, :])
        local_index = corner_local_nodes_non_dirichlet[i]
        local_matrix = local_matrix.at[local_index, local_index].set(0)
        global_nodes = elements[corner_elements[i], :]
        for row in range(4):
            for col in range(4):
                K = K.at[global_nodes[row], global_nodes[col]].add(local_matrix[row,col])
    return K

def assemble_stiffness_dir_edges(edge_elements, elements, n_nodes, edge_local_nodes_non_dirichlet, edge_length):
    K = jnp.zeros((n_nodes, n_nodes))
    for i in range(edge_local_nodes_non_dirichlet.shape[0]):
        local_matrix = element_stiffness(edge_length[i, :])
        local_indices = edge_local_nodes_non_dirichlet[i, :]
        for row in local_indices:
            for col in local_indices:
                local_matrix = local_matrix.at[row, col].set(0)
        global_nodes = elements[edge_elements[i], :]
        for row in range(4):
            for col in range(4):
                K = K.at[global_nodes[row], global_nodes[col]].add(local_matrix[row,col])
    return K
def assemble_stiffness_dir(element_dirichlet, elements, n_nodes, element_dir_length):
    # Compute stiffness matrices for all elements
    # element_stiffness_matrices = jax.vmap(element_stiffness_dir)(mask_and_length)
    K = jnp.zeros((n_nodes, n_nodes))

    indices = jnp.arange(4)
    for i in range(element_dirichlet.shape[0]):
        local_matrix = element_stiffness(element_dir_length[i, :])
        # lend = element_dirichlet[i, 5]
        local_indices = element_dirichlet[i, 1:]
        local_indices = local_indices[local_indices != -1]
        # local_matrix = local_matrix.at[local_indices, :].set(0)
        # local_matrix = local_matrix.at[:, local_indices].set(0)
        print(local_indices)
        for row in local_indices:
            for col in local_indices:
                local_matrix = local_matrix.at[row, col].set(0)
        # print(local_matrix)
        global_nodes = elements[element_dirichlet[i, 0], :]
        # print(global_nodes)
        # print(global_nodes.shape)
        # print(local_matrix.shape)
        for row in range(4):
            for col in range(4):
                K = K.at[global_nodes[row], global_nodes[col]].add(local_matrix[row,col])

    # print(element_stiffness_matrices)
    
    # Generate row and column indices for scatter addition
    # row_indices = elements[:, None, :].repeat(elements.shape[1], axis=1)
    # col_indices = elements[:, :, None].repeat(elements.shape[1], axis=2)
    
    # Use scatter addition to accumulate values into the global matrix
    # K = K.at[row_indices, col_indices].add(element_stiffness_matrices)

    return K


# Assemble the load vector
# @jit
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
        sum1 = jnp.sum(f_vals * values_phi1_ * wx * wy * jacobian)
        sum2 = jnp.sum(f_vals * values_phi2_ * wx * wy * jacobian)
        sum3 = jnp.sum(f_vals * values_phi3_ * wx * wy * jacobian)

        return elements[e, :], jnp.array([sum0, sum1, sum2, sum3])

    F = jnp.zeros((coords.shape[0]))
    n_el = elements.shape[0]
    element_indices, element_contributions = jax.vmap(compute_element)(jnp.arange(n_el))
    F = F.at[element_indices].add(element_contributions)

    return F


# Apply boundary conditions
# @jit
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

    return K, F

# def apply_boundary_conditions(K, F, dirichlet_nodes, neumann_edges, elements, coords):
#     problem_test = problem(problem_number)
#     K = K.at[dirichlet_nodes, :].set(0)
#     K = K.at[:, dirichlet_nodes].set(0)
#     K = K.at[dirichlet_nodes, dirichlet_nodes].set(1)
#     aux1 = 2*jnp.sqrt(10/7)
#     aux2 = 13*jnp.sqrt(70)
#     nodes = jnp.array([-1/3*jnp.sqrt(5+aux1), -1/3*jnp.sqrt(5-aux1), 0, 1/3*jnp.sqrt(5-aux1), 1/3*jnp.sqrt(5+aux1)])
#     weights = jnp.array([(322-aux2)/900, (322+aux2)/900, 128/225, (322+aux2)/900, (322-aux2)/900])
    
#     for i in range(neumann_edges.shape[0]//2):
#     # for i in range(neumann_edges.shape[0]//3):
#     # for i in range(1):
#         g = problem_test.gr
#         e = neumann_edges[i, 0]
#         n1 = neumann_edges[i, 1]
#         n2 = neumann_edges[i, 2]
#         e = e.astype(jnp.int64)
#         n1 = n1.astype(jnp.int64)
#         n2 = n2.astype(jnp.int64)
#         x1, y1 = coords[elements[e, n1], :]
#         x2, y2 = coords[elements[e, n2], :]
#         norm = 0.5 * jnp.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
#         transformed_x = x1 + 0.5 * (nodes + 1) * (x2 - x1)
#         transformed_y = y1 + 0.5 * (nodes + 1) * (y2 - y1)
#         sum0 = jnp.sum(norm * weights * g(transformed_x, transformed_y) * (0.5 * (1 - nodes)))
#         sum1 = jnp.sum(norm * weights * g(transformed_x, transformed_y) * (0.5 * (1 + nodes)))

#         F = F.at[elements[e, n1]].add(sum0)
#         F = F.at[elements[e, n2]].add(sum1)

#     for i in range(neumann_edges.shape[0]//2, neumann_edges.shape[0]):
#     # for i in range(neumann_edges.shape[0]//3, neumann_edges.shape[0]):
#     # for i in range(1, neumann_edges.shape[0]):
#         g = problem_test.gu
#         e = neumann_edges[i, 0]
#         n1 = neumann_edges[i, 1]
#         n2 = neumann_edges[i, 2]
#         e = e.astype(jnp.int64)
#         n1 = n1.astype(jnp.int64)
#         n2 = n2.astype(jnp.int64)
#         x1, y1 = coords[elements[e, n1], :]
#         x2, y2 = coords[elements[e, n2], :]
#         norm = 0.5 * jnp.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
#         transformed_x = x1 + 0.5 * (nodes + 1) * (x2 - x1)
#         transformed_y = y1 + 0.5 * (nodes + 1) * (y2 - y1)
#         sum0 = jnp.sum(norm * weights * g(transformed_x, transformed_y) * (0.5 * (1 - nodes)))
#         sum1 = jnp.sum(norm * weights * g(transformed_x, transformed_y) * (0.5 * (1 + nodes)))

#         F = F.at[elements[e, n1]].add(sum0)
#         F = F.at[elements[e, n2]].add(sum1)

#     F = F.at[dirichlet_nodes].set(0)
#     return K, F

# Solve the system
# @jit
def solve(theta):
    nx = int(theta.shape[1]/2) + 1
    ny = nx
    # print(ny, nx)
    node_coords_x, node_coords_y  = softmax_nodes(theta)
    # node_coords_x = jnp.linspace(0, 1, nx)
    # node_coords_y = jnp.linspace(0, 1, ny)
    coords, elements = generate_mesh(nx, ny, node_coords_x, node_coords_y)
    n_elements = elements.shape[0]
    n_nodes = coords.shape[0]

    # dirichlet_nodes = jnp.append(jnp.arange(nx),nx*jnp.arange(1,ny))
    dirichlet_nodes = jnp.concatenate([jnp.arange(nx),nx*jnp.arange(1,ny),nx*jnp.arange(1,ny)+nx-1, nx*(ny-1)+jnp.arange(1,nx-1)])
    # dirichlet_nodes = nx*jnp.arange(ny)
    
    corner_elements = jnp.array([0, nx-2, (nx-1)*(ny-2), (nx-1)*(ny-1)-1])
    corner_local_nodes_non_dirichlet = jnp.array([2,3,1,0])

    edge_elements = jnp.concatenate([jnp.arange(1, nx-2), (nx-1)*jnp.arange(1, ny-2), (nx-1)*jnp.arange(1, ny-2) + (nx-2), (nx-1)*(ny-2) + jnp.arange(1, nx-2)])
    ind1 = jnp.concatenate([2*jnp.ones(nx-3), jnp.ones(ny-3), jnp.zeros(ny-3), jnp.zeros(nx-3)])
    ind2 = jnp.concatenate([3*jnp.ones(nx-3), 2*jnp.ones(ny-3), 3*jnp.ones(ny-3), jnp.ones(nx-3)])
    edge_local_nodes_non_dirichlet = jnp.concatenate([ind1,ind2]).reshape((2*(nx-3)+2*(ny-3),2), order='F')
    # print(edge_local_nodes_non_dirichlet)
    edge_local_nodes_non_dirichlet = edge_local_nodes_non_dirichlet.astype(int)

    corner_length = jnp.zeros((4,2))
    for i in range(4):
        e = corner_elements[i]
        x1, y1 = coords[elements[e, 0], :]
        x2, y2 = coords[elements[e, 2], :]
        corner_length = corner_length.at[i,0].set(x2-x1)
        corner_length = corner_length.at[i,1].set(y2-y1)

    # print(edge_elements.shape)
    edge_length = jnp.zeros((edge_elements.shape[0],2))
    for i in range(edge_elements.shape[0]):
        e = edge_elements[i]
        x1, y1 = coords[elements[e, 0], :]
        x2, y2 = coords[elements[e, 2], :]
        edge_length = edge_length.at[i,0].set(x2-x1)
        edge_length = edge_length.at[i,1].set(y2-y1)
    
    K_dir_corners = assemble_stiffness_dir_corners(corner_elements, elements, n_nodes, corner_local_nodes_non_dirichlet, corner_length)
    K_dir_edges = assemble_stiffness_dir_edges(edge_elements, elements, n_nodes, edge_local_nodes_non_dirichlet, edge_length)

    # print('aaaa')

    # print(dirichlet_nodes)
    # Extract the coordinates for the start and end points of each element
    start_coords = coords[elements[:, 0], :]
    end_coords = coords[elements[:, 2], :]

    # Compute element lengths in a vectorized manner
    element_length = end_coords - start_coords

    K = assemble_stiffness(n_elements, elements, element_length, n_nodes)

    identity_matrix = jnp.eye(n_nodes)

    # print('alo')
    # Create a mask for dirichlet_nodes
    dirichlet_mask = jnp.zeros((n_nodes, n_nodes))
    dirichlet_mask = dirichlet_mask.at[dirichlet_nodes, dirichlet_nodes].set(1)

    # Combine the identity matrix and the dirichlet_mask
    I_dir = identity_matrix * dirichlet_mask


    F = load_vector(coords, elements)
    Knew = K - K_dir_corners - K_dir_edges + I_dir

    K, F = apply_boundary_conditions(K, F, dirichlet_nodes)
    # K, F = apply_boundary_conditions(K, F, dirichlet_nodes, neumann_edges, elements, coords)
    # print(Knew)
    # print('------------')
    # print(K)
    # print(jnp.linalg.norm(K-Knew))
    u = jnp.linalg.solve(Knew, F)

    return coords, u

# Solve the system
# @jit
def solve_and_loss(theta):
    nx = int(theta.shape[1]/2) + 1
    ny = nx
    # print(ny, nx)
    node_coords_x, node_coords_y  = softmax_nodes(theta)
    # node_coords_x = jnp.linspace(0, 1, nx)
    # node_coords_y = jnp.linspace(0, 1, ny)
    coords, elements = generate_mesh(nx, ny, node_coords_x, node_coords_y)
    n_elements = elements.shape[0]
    n_nodes = coords.shape[0]

    # dirichlet_nodes = jnp.append(jnp.arange(nx),nx*jnp.arange(1,ny))
    dirichlet_nodes = jnp.concatenate([jnp.arange(nx),nx*jnp.arange(1,ny),nx*jnp.arange(1,ny)+nx-1, nx*(ny-1)+jnp.arange(1,nx-1)])
    # dirichlet_nodes = nx*jnp.arange(ny)
    
    corner_elements = jnp.array([0, nx-2, (nx-1)*(ny-2), (nx-1)*(ny-1)-1])
    corner_local_nodes_non_dirichlet = jnp.array([2,3,1,0])

    edge_elements = jnp.concatenate([jnp.arange(1, nx-2), (nx-1)*jnp.arange(1, ny-2), (nx-1)*jnp.arange(1, ny-2) + (nx-2), (nx-1)*(ny-2) + jnp.arange(1, nx-2)])
    ind1 = jnp.concatenate([2*jnp.ones(nx-3), jnp.ones(ny-3), jnp.zeros(ny-3), jnp.zeros(nx-3)])
    ind2 = jnp.concatenate([3*jnp.ones(nx-3), 2*jnp.ones(ny-3), 3*jnp.ones(ny-3), jnp.ones(nx-3)])
    edge_local_nodes_non_dirichlet = jnp.concatenate([ind1,ind2]).reshape((2*(nx-3)+2*(ny-3),2), order='F')
    # print(edge_local_nodes_non_dirichlet)
    edge_local_nodes_non_dirichlet = edge_local_nodes_non_dirichlet.astype(int)

    corner_length = jnp.zeros((4,2))
    for i in range(4):
        e = corner_elements[i]
        x1, y1 = coords[elements[e, 0], :]
        x2, y2 = coords[elements[e, 2], :]
        corner_length = corner_length.at[i,0].set(x2-x1)
        corner_length = corner_length.at[i,1].set(y2-y1)

    # print(edge_elements.shape)
    edge_length = jnp.zeros((edge_elements.shape[0],2))
    for i in range(edge_elements.shape[0]):
        e = edge_elements[i]
        x1, y1 = coords[elements[e, 0], :]
        x2, y2 = coords[elements[e, 2], :]
        edge_length = edge_length.at[i,0].set(x2-x1)
        edge_length = edge_length.at[i,1].set(y2-y1)
    
    K_dir_corners = assemble_stiffness_dir_corners(corner_elements, elements, n_nodes, corner_local_nodes_non_dirichlet, corner_length)
    K_dir_edges = assemble_stiffness_dir_edges(edge_elements, elements, n_nodes, edge_local_nodes_non_dirichlet, edge_length)

    # print('aaaa')

    # print(dirichlet_nodes)
    # Extract the coordinates for the start and end points of each element
    start_coords = coords[elements[:, 0], :]
    end_coords = coords[elements[:, 2], :]

    # Compute element lengths in a vectorized manner
    element_length = end_coords - start_coords

    K = assemble_stiffness(n_elements, elements, element_length, n_nodes)

    identity_matrix = jnp.eye(n_nodes)

    # print('alo')
    # Create a mask for dirichlet_nodes
    dirichlet_mask = jnp.zeros((n_nodes, n_nodes))
    dirichlet_mask = dirichlet_mask.at[dirichlet_nodes, dirichlet_nodes].set(1)

    # Combine the identity matrix and the dirichlet_mask
    I_dir = identity_matrix * dirichlet_mask


    F = load_vector(coords, elements)
    Knew = K - K_dir_corners - K_dir_edges + I_dir

    K, F = apply_boundary_conditions(K, F, dirichlet_nodes)
    # K, F = apply_boundary_conditions(K, F, dirichlet_nodes, neumann_edges, elements, coords)
    # print(Knew)
    # print('------------')
    # print(K)
    # print(jnp.linalg.norm(K-Knew))
    u = jnp.linalg.solve(Knew, F)

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
    ]

    # Ensure problem_number is within bounds
    if problem_number < 0 or problem_number >= len(problems_data):
        raise ValueError(f"Invalid problem number: {problem_number}. Must be between 0 and {len(problems_data) - 1}.")

    # Retrieve problem data and unpack into the Elliptic1D class
    data = problems_data[problem_number]
    return Elliptic1D(**data)


# print(solve_and_loss(jnp.ones((100))))
# coords, u = solve(jnp.ones((1,50)))
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
# plt.savefig('femtest.png')
# plt.show()