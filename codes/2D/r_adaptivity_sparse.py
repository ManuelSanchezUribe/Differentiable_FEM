########################################################################################
# Workshop: Coding for PDEs with Neural Networks
# Date: 2025-24-01
# Author: Danilo Aballay, Vicente Iligaray, Ignacio Tapia y Manuel Sánchez
########################################################################################

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import os

os.environ["KERAS_BACKEND"] = "jax"

import keras

from Laplace_JAXSparse2D import solve_and_loss, solve

# Set the random seed
np.random.seed(1234)
keras.utils.set_random_seed(1234)

dtype='float64' # double precision set to default in the SCR functions
jax.config.update("jax_enable_x64", True)
keras.backend.set_floatx(dtype)


# =============================================================================
#
#          Source code - PINNs H01 1D
#
# =============================================================================

class special_layer(keras.layers.Layer):
    def __init__(self, n_nodes, dimension, w_interior_initial_values = None, **kwargs):
        super().__init__(**kwargs)

        self.mobile_interior_vertices = self.add_weight(shape = (dimension, n_nodes), initializer = 'ones')
        
        if w_interior_initial_values is not None:
            self.mobile_interior_vertices.assign(w_interior_initial_values)
        else:
            self.mobile_interior_vertices.assign(jnp.ones((dimension, n_nodes)) / n_nodes)

    def call(self, inputs):
        return jnp.array(self.mobile_interior_vertices)

def make_model(n_nodes, dimension=1, w_interior_initial_values=None):
    L = special_layer(n_nodes, dimension, w_interior_initial_values)
    xvals = keras.layers.Input(shape=(1,), name='x_input',dtype=dtype)
    output = L(xvals)
    model = keras.Model(inputs=xvals, outputs=output, name='model')
    return model

##PINNs loss function ( loss into layer )
class loss(keras.layers.Layer):
    def __init__(self,model,**kwargs):

        """
        Initializes the PINNS loss layer with provided parameters.

        Args:
            model (keras.Model): The neural network model for the approximate
                                    solution.
            n_pts (int): Number of integration points.
            f (function): Source - RHS of the PDE

            kwargs: Additional keyword arguments.
        """
        super(loss, self).__init__()

        self.model = model

    def call(self, inputs):

        """
        Computes the collocation - PINNs loss.

        Args:
            inputs: The input data (dummy).

        Returns:
            keras.Tensor: The loss value.
        """

        theta = self.model(jnp.array([1]))
        loss = solve_and_loss(theta)
        return loss


## Create a loss model
def make_loss_model(model):
    """
    Constructs a loss model for PINNs.

    Args:
        model (keras.Model): The neural network model for the approximate solution.
        n_pts (int): Number of integration points.

    Returns:
        keras.Model: A model with the collocation-based loss function.
    """
    xvals = keras.layers.Input(shape=(1,), name='x_input',dtype=dtype)

    # Compute the loss using the provided neural network and
    # integration parameters
    output = loss(model)(xvals)
    # output = loss_dummy(model)(xvals)
    # Create a Keras model for the loss
    loss_model = keras.Model(inputs=xvals, outputs=output)

    return loss_model


def tricky_loss(y_pred, y_true):
    """
    A placeholder loss function that can be replaced as needed.

    Args:
        y_pred: Predicted values.
        y_true: True values.

    Returns:
        float: The loss value.
    """
    # This is a placeholder loss function that can be substituted with a
    # custom loss if required.
    return y_true

# =============================================================================
#
#          Example 1 - Inputs
#
# =============================================================================


# Number of neurons per hidden layer in the neural network
nn = int(2 * 2**4) # Two times the number of neurons 

# Number of training iterations
iterations = 1000

# Initialize the neural network model for the approximate solution
model = make_model(nn)

init_nodes = model(jnp.array([1]))

# Big model including the  loss
loss_model = make_loss_model(model)

# Optimizer (Adam optimizer with a specific learning rate)
optimizer = keras.optimizers.Adam(learning_rate=1e-2)

# Adatative learning rate
def lr_schedule(epoch, lr):
    if epoch >= 5000:
        return 1e-3
    return lr
lr_scheduler = keras.callbacks.LearningRateScheduler(lr_schedule)

# Compile the loss model with a custom loss function (tricky_loss)
loss_model.compile(optimizer=optimizer, loss=tricky_loss)

# Train the model using a single training data point ([1.], [1.]) for a
# specified number of epochs (iterations)
history = loss_model.fit(jnp.array([1.]), jnp.array([1.]), epochs=iterations, callbacks = [lr_scheduler])

# #Plot loss history
plt.figure()
plt.plot(history.history['loss'])
# plt.savefig('loss.png')

# # node_coords, u = solve(theta)

node_coords, u = solve(model(jnp.array([1])))
init_coords, o = solve(init_nodes)

# print(node_coords)

# # # Output results
# print("Node coordinates:", node_coords)
# # print("Solution u:", u)
# # print(val)


# ## ---------
# # SOLUTION
## ---------

# Obtener dimensiones de la grilla
unique_x = np.unique(node_coords[:, 0])  # Valores únicos de x
unique_y = np.unique(node_coords[:, 1])  # Valores únicos de y

# Reorganizar Z en una matriz 2D
Z_grid = u.reshape(len(unique_y), len(unique_x))  # Debe coincidir con la estructura de la grilla

# Crear grilla a partir de coords
X, Y = np.meshgrid(unique_x, unique_y)

# Graficar la grilla con degradado entre nodos
plt.figure(figsize=(8, 6))
plt.pcolormesh(X, Y, Z_grid, shading='gouraud', cmap='viridis')  # shading='gouraud' para degradados suaves
plt.colorbar(label='u (Solución)')

# Añadir las líneas de la malla
for xi in unique_x:
    plt.plot([xi] * len(unique_y), unique_y, color='black', linewidth=0.5)  # Líneas verticales
for yi in unique_y:
    plt.plot(unique_x, [yi] * len(unique_x), color='black', linewidth=0.5)  # Líneas horizontales

# Configuración del gráfico
plt.title('Resultados')
plt.xlabel('x')
plt.ylabel('y')
# plt.axis('equal')  # Mantener proporciones
plt.show()