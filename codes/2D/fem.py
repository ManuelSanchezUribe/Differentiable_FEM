import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve
from jax import config; config.update("jax_enable_x64", True)

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
    F = jnp.zeros((coords.shape[0]))
    n_el = elements.shape[0]
    phi0 = lambda xi, eta: 0.25*(1-xi)*(1-eta)
    phi1 = lambda xi, eta: 0.25*(1+xi)*(1-eta)
    phi2 = lambda xi, eta: 0.25*(1+xi)*(1+eta)
    phi3 = lambda xi, eta: 0.25*(1-xi)*(1+eta)
    for e in range(n_el):
        x1,y1 = coords[elements[e,0],:]
        x2,y2 = coords[elements[e,2],:]
        hx = x2-x1; hy = y2-y1
        transf_nodes_x = 0.5*hx*nodes + 0.5*(x1+x2)
        transf_nodes_y = 0.5*hy*nodes + 0.5*(y1+y2)
        sum0 = 0
        sum1 = 0
        sum2 = 0
        sum3 = 0
        for i in range(5):
            for j in range(5):
                sum0 += f(transf_nodes_x[i], transf_nodes_y[j]) * phi0(nodes[i],nodes[j]) * weights[i] * weights[j] * hx * hy * 0.25
                sum1 += f(transf_nodes_x[i], transf_nodes_y[j]) * phi1(nodes[i],nodes[j]) * weights[i] * weights[j] * hx * hy * 0.25
                sum2 += f(transf_nodes_x[i], transf_nodes_y[j]) * phi2(nodes[i],nodes[j]) * weights[i] * weights[j] * hx * hy * 0.25
                sum3 += f(transf_nodes_x[i], transf_nodes_y[j]) * phi3(nodes[i],nodes[j]) * weights[i] * weights[j] * hx * hy * 0.25
        F = F.at[elements[e,:]].add([sum0,sum1,sum2,sum3])
    
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
print(elements)
f = lambda x,y: 1
F = load_vector(coords, elements, f)
print(F)
print(jnp.sum(F))

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
