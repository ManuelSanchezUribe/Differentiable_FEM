import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve
from jax import config; config.update("jax_enable_x64", True)

values_phi0_ = jnp.array([[0.9083804012656871,  0.7331497981296533,  0.47654496148466596, 0.21994012483967862, 0.04470952170364481],
                         [0.7331497981296533,  0.591721954534264, 0.38461732752642075, 0.17751270051857745,  0.036084856923188136],
                         [0.47654496148466596, 0.38461732752642075, 0.25,       0.11538267247357925, 0.02345503851533401],
                         [0.21994012483967862, 0.17751270051857745,  0.11538267247357925, 0.053252644428581054, 0.010825220107479883],
                         [0.04470952170364481, 0.036084856923188136, 0.02345503851533401, 0.010825220107479883, 0.002200555327023207]])
# jnp.set_printoptions(precision=None, threshold =10000000000)

# Generate a structured grid
def generate_mesh(nx, ny, x_min, x_max, y_min, y_max):
    x = jnp.linspace(x_min, x_max, nx)
    y = jnp.linspace(y_min, y_max, ny)
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

# Define basis function gradients in the reference element
def grad_phi():
    # Gradients of basis functions in reference quadrilateral
    return jnp.array([
        [-0.25, -0.25], [0.25, -0.25],
        [0.25, 0.25], [-0.25, 0.25]
    ])

# Assemble the stiffness matrix
def stiffness_matrix(nodes, elements):
    grad_N = grad_phi()  # Shape: (4, 2)
    element_nodes = nodes[elements]  # Shape: (n_elements, 4, 2)

    # Compute Jacobians and their determinants
    J = jnp.einsum("ki,eij->ekj", grad_N, element_nodes)  # Shape: (n_elements, 2, 2)
    detJ = jnp.linalg.det(J)  # Shape: (n_elements,)
    invJ = jnp.linalg.inv(J)  # Shape: (n_elements, 2, 2)
    grad_N_mapped = jnp.einsum("ei,ejk->eik", grad_N, invJ)  # Shape: (n_elements, 4, 2)

    # Element stiffness matrices
    element_stiffness = detJ * jnp.einsum("eik,ejk->eij", grad_N_mapped, grad_N_mapped)

    # Assemble global stiffness matrix
    n_nodes = nodes.shape[0]
    indices = jnp.stack(jnp.meshgrid(elements, elements, indexing="ij"), axis=-1).reshape(-1, 2)
    values = element_stiffness.flatten()
    A = jax.ops.segment_sum(values, indices[:, 0] * n_nodes + indices[:, 1], n_nodes ** 2)
    A = A.reshape((n_nodes, n_nodes))
    return A


# Assemble the load vector
def load_vector(coords, elements, f):
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

# Apply Dirichlet boundary conditions
def apply_dirichlet(A, b, boundary_nodes, boundary_values):
    A = A.at[boundary_nodes].set(0)
    A = A.at[:, boundary_nodes].set(0)
    A = A.at[boundary_nodes, boundary_nodes].set(1)
    b = b.at[boundary_nodes].set(boundary_values)
    return A, b

# Main solver function
def solve_fem_2d(nx, ny, x_min, x_max, y_min, y_max, f, boundary_nodes, boundary_values):
    X, Y, (nx, ny) = generate_mesh(nx, ny, x_min, x_max, y_min, y_max)
    nodes = jnp.stack([X.ravel(), Y.ravel()], axis=1)

    # Create quadrilateral elements
    i, j = jnp.meshgrid(jnp.arange(nx - 1), jnp.arange(ny - 1), indexing="ij")
    n0 = i * ny + j
    n1 = n0 + 1
    n2 = n0 + ny
    n3 = n2 + 1
    elements = jnp.stack([n0, n1, n3, n2], axis=-1).reshape(-1, 4)

    # Assemble system
    A = stiffness_matrix(nodes, elements)
    b = load_vector(nodes, elements, f)
    A, b = apply_dirichlet(A, b, boundary_nodes, boundary_values)

    # Solve the linear system
    u = solve(A, b)
    return X, Y, u.reshape((nx, ny))


nx, ny = 5, 5
x_min, x_max = 0.0, 1.0
y_min, y_max = 0.0, 1.0
coords, elements = generate_mesh(nx, ny, x_min, x_max, y_min, y_max)
dirichlet_nodes = jnp.append(jnp.arange(nx),nx*jnp.arange(1,ny))
neumann_nodes = jnp.append(nx*jnp.arange(1,ny)-1, jnp.arange((ny-1)*nx, ny*nx))
# print(elements)
f = lambda x,y: x*y
F = load_vector(coords, elements, f)
print(dirichlet_nodes)
print(neumann_nodes)
# print(F)
# print(jnp.sum(F))


# # Example usage
# if __name__ == "__main__":
#     nx, ny = 20, 20
#     x_min, x_max = 0.0, 1.0
#     y_min, y_max = 0.0, 1.0

#     def f(x):
#         return 1.0  # Source term (e.g., uniform)

#     boundary_nodes = jnp.array([0, nx - 1])  # Example: corner nodes
#     boundary_values = jnp.array([0.0, 1.0])  # Boundary values

#     X, Y, u = solve_fem_2d(nx, ny, x_min, x_max, y_min, y_max, f, boundary_nodes, boundary_values)

#     # Visualization (optional)
#     import matplotlib.pyplot as plt
#     plt.contourf(X, Y, u, levels=50)
#     plt.colorbar()
#     plt.title("FEM Solution to Laplace Equation (Quadrilateral Elements)")
#     plt.show()
