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
    rows = jnp.zeros((1,n_nodes+1))
    cols = jnp.zeros((1,lenval))




    element_nodes = jnp.array([[i, i + 1] for i in range(n_elements)])
    coords = node_coords[element_nodes]
    h_values = element_length

    ke_values = jax.vmap(element_stiffness)(h_values)

    for i in range(n_elements):
        values = values.at[0, (3*(i)):(3*(i)+4)].add(ke_values[i,:,:].flatten())

    # fe_values = jax.vmap(element_load)(coords)

    # K = jnp.zeros((n_nodes, n_nodes))
    # F = jnp.zeros(n_nodes)

    # K = K.at[element_nodes[:, 0], element_nodes[:, 0]].add(ke_values[:, 0, 0])
    # K = K.at[element_nodes[:, 0], element_nodes[:, 1]].add(ke_values[:, 0, 1])
    # K = K.at[element_nodes[:, 1], element_nodes[:, 0]].add(ke_values[:, 1, 0])
    # K = K.at[element_nodes[:, 1], element_nodes[:, 1]].add(ke_values[:, 1, 1])

    # F = F.at[element_nodes[:, 0]].add(fe_values[:, 0])
    # F = F.at[element_nodes[:, 1]].add(fe_values[:, 1])

    return values

print(assemble(4,jnp.array([0,0.3,0.5,0.9,1.0], dtype=jnp.float64),jnp.array([0.3,0.2,0.4,0.1], dtype=jnp.float64), 5))