########################################################################################
# Workshop: Coding for PDEs with Neural Networks
# Date: 2025-24-01
# Author: Danilo Aballay, Vicente Iligaray, Ignacio Tapia y Manuel Sánchez
########################################################################################

import jax
import jax.numpy as jnp
from jax import jit
from jax import config; config.update("jax_enable_x64", True)

# Element stiffness matrix and load vector
def element_stiffness(h):
    return jnp.array([[1, -1], [-1, 1]], dtype=jnp.float64) / h

# def element_load(coords):
#     x1, x2 = coords
#     p1 = -1/jnp.sqrt(3)
#     p2 = 1/jnp.sqrt(3)
#     pt1 = (x2 - x1) * p1 / 2 + (x2 + x1) / 2
#     pt2 = (x2 - x1) * p2 / 2 + (x2 + x1) / 2
#     phiatpt1 = (p2+1)/2
#     phiatpt2 = (1+p1)/2
#     #midpoint = (x1 + x2) / 2
#     h = x2 - x1
#     problem_test = problem(problem_number)
#     f = problem_test.f
#     return h * jnp.array([f(pt1)*phiatpt1 + f(pt2)*phiatpt2, f(pt1)*phiatpt2 + f(pt2)*phiatpt1], dtype=jnp.float64) / 2

def assemble(n_elements, node_coords, element_length, n_nodes):
    lenval = 3*(n_nodes-2) + 4
    values = jnp.zeros((1,lenval))
    # rows =  jnp.append(3 * jnp.arange(n_nodes) - jnp.arange(n_nodes) % 2,n_nodes*3-1)
    rows = 3 * jnp.arange(n_nodes+1) - 1
    rows = rows.at[0].set(0)
    rows = rows.at[-1].add(-1)
    # cols = jnp.append(jnp.concatenate([jnp.arange(max([0,i-1]),min(i+2, n_nodes)) for i in range(n_nodes-1)]),jnp.array([n_nodes-2,n_nodes-1]))

    start_indices = jnp.maximum(0, jnp.arange(n_nodes - 1) - 1)
    end_indices = jnp.minimum(jnp.arange(n_nodes - 1) + 2, n_nodes)

    # Compute ranges for all elements at once
    max_range = 3  # Maximum possible range size (i.e., 3 elements: i-1, i, i+1)
    range_matrix = jnp.arange(max_range) + start_indices[:, None]  # Broadcast addition
    valid_mask = (range_matrix < end_indices[:, None]) & (range_matrix >= start_indices[:, None])

    # Mask out invalid elements and flatten
    valid_values = jnp.where(valid_mask, range_matrix, -1)
    # print(valid_values)
    # print(valid_values[1:])
    # concatenated = valid_values[valid_values != -1]  # Remove invalid (-1) elements

    # Append the final elements
    cols = jnp.array([0,1])
    cols = jnp.append(cols, valid_values[1:])
    cols = jnp.append(cols, jnp.array([n_nodes - 2, n_nodes - 1]))


    element_nodes = jnp.array([[i, i + 1] for i in range(n_elements)])
    coords = node_coords[element_nodes]
    h_values = element_length

    ke_values = jax.vmap(element_stiffness)(h_values)

    # Compute the indices for vectorized updates
    indices = jnp.arange(n_elements)  # Array of element indices
    flat_indices = (3 * indices[:, None] + jnp.arange(4)).reshape(-1)  # Compute flattened indices

    # Flatten ke_values to match flat_indices
    flattened_ke_values = ke_values.reshape(-1)

    # Perform the update in a vectorized manner
    values = values.at[0, flat_indices].add(flattened_ke_values)

    # for i in range(n_elements):
    #     values = values.at[0, (3*(i)):(3*(i)+4)].add(ke_values[i,:,:].flatten())

    # fe_values = jax.vmap(element_load)(coords)

    # K = jnp.zeros((n_nodes, n_nodes))
    # F = jnp.zeros(n_nodes)

    # K = K.at[element_nodes[:, 0], element_nodes[:, 0]].add(ke_values[:, 0, 0])
    # K = K.at[element_nodes[:, 0], element_nodes[:, 1]].add(ke_values[:, 0, 1])
    # K = K.at[element_nodes[:, 1], element_nodes[:, 0]].add(ke_values[:, 1, 0])
    # K = K.at[element_nodes[:, 1], element_nodes[:, 1]].add(ke_values[:, 1, 1])

    # F = F.at[element_nodes[:, 0]].add(fe_values[:, 0])
    # F = F.at[element_nodes[:, 1]].add(fe_values[:, 1])

    return rows, values, cols

print(assemble(4,jnp.array([0,0.3,0.5,0.9,1.0], dtype=jnp.float64),jnp.array([0.3,0.2,0.4,0.1], dtype=jnp.float64), 5))