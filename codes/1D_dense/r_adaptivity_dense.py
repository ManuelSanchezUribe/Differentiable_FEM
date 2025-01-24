# -*- coding: utf-8 -*-

# This code presents a simple implementation of Physics-Informed
# Neural Networks (PINNs) as a collocation method. -- <50 lines of PINNS--

# In this 1D example, we utilize Keras for constructing neural networks
# and JAX in the backend.


import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import os
import time

os.environ["KERAS_BACKEND"] = "jax"

import keras

from Laplace_JAXDense import solve_and_loss, solve

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

def make_special_model(n_nodes, dimension=1, w_interior_initial_values=None):
    L = special_layer(n_nodes, dimension, w_interior_initial_values)
    xvals = keras.layers.Input(shape=(1,), name='x_input',dtype=dtype)
    output = L(xvals)
    model = keras.Model(inputs=xvals, outputs=output, name='model')
    return model

## Define an approximate solution (u_nn): A neural network model
def make_model(neurons, n_layers, n_output, activation='tanh'):

    """
    Creates a neural network model to approximate the solution of
        int (grad u ).(grad v) - int f.v = 0

    Args:
        neurons (int): The number of neurons in each hidden layer.
        activation (str, optional): Activation function for hidden layers.

    Returns:
        keras.Model: A neural network model for the approximate solution.
    """

    xvals = keras.layers.Input(shape=(1,), name='x_input',dtype=dtype)

	# The input
    l1 = keras.layers.Dense(neurons, activation=activation, dtype=dtype)(xvals)

    ## ---------------
    #  The dense layers
    ## ---------------

    # First layer
    for l in range(n_layers-2):
        # Hidden layers
        l1 = keras.layers.Dense(neurons, activation=activation, dtype=dtype)(l1)
    # Last layer
    output = keras.layers.Dense(n_output, activation=activation, dtype=dtype)(l1)

    model = keras.Model(inputs = xvals, outputs = output, name='model')

    # Print the information of the model u
    # model.summary()

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
nn = 100000

# Number of training iterations
iterations = 10000

# Initialize the neural network model for the approximate solution
model = make_special_model(nn)

init_nodes = model(jnp.array([1]))

# Big model including the  loss
loss_model = make_loss_model(model)

# Optimizer (Adam optimizer with a specific learning rate)
optimizer = keras.optimizers.Adam(learning_rate=1e-3)

# Adatative learning rate
def lr_schedule(epoch, lr):
    if epoch >= 5000:
        return 1e-4
    return lr
lr_scheduler = keras.callbacks.LearningRateScheduler(lr_schedule)

# Compile the loss model with a custom loss function (tricky_loss)
loss_model.compile(optimizer=optimizer, loss=tricky_loss)

# Train the model
start_time = time.time()
history    = loss_model.fit(jnp.array([1.]), jnp.array([1.]), epochs=iterations, callbacks = [lr_scheduler])
end_time   = time.time() 
print('Training time: ', end_time - start_time)

# Plot loss history
plt.figure(1)
plt.plot(history.history['loss'])
plt.xlabel('Epoch')
plt.ylabel('Ritz')

# Aumentar la resolución del eje y
min_loss = min(history.history['loss'])
max_loss = max(history.history['loss'])
ticks = np.linspace(min_loss, max_loss, num=5)  # Ajusta el número de ticks según sea necesario
plt.yticks(ticks)
plt.grid(which='both', axis='both', linestyle=':', color='gray')
plt.tight_layout()
plt.title('Loss history')
plt.savefig('../Figures/Loss_history' + str(nn) + '_iter' + str(iterations) + '.png')

node_coords, u = solve(model(jnp.array([1])))
init_coords, o = solve(init_nodes)

# ## ---------
# # SOLUTION
## ---------

# fig, ax = plt.subplots()
# # Plot the approximate solution obtained from the trained model
plt.figure(2)
plt.plot(node_coords, u,'o--', color='b')
# plt.plot(init_coords, np.zeros(len(node_coords)),'o', color='k', alpha=0.5, markersize=5)
# plt.plot(node_coords, np.zeros(len(node_coords)),'o', color='r', alpha=0.5, markersize=5)
# plt.legend(['u', 'nodes', 'initial nodes'])
plt.xlabel('x')
plt.ylabel('u')
plt.title('Results')
plt.grid(which = 'both', axis = 'both', linestyle = ':', color = 'gray')
plt.tight_layout()

# Guardar la figura en el directorio Figures
plt.savefig('../Figures/r_adaptivity_sparse_nodes' + str(nn) + '_iter' + str(iterations) + '.png')


# Plot Grid Mesh

# Crear la figura
plt.figure(3)
# Graficar la grilla con espaciamiento variable
plt.plot(init_coords, np.zeros(len(node_coords)),'o', color='k', alpha=0.5, markersize=10)
plt.plot(node_coords, np.zeros(len(node_coords)),'o', color='r', alpha=0.6, markersize=10)

# Añadir etiquetas y leyenda
plt.xlabel('Nodes')
plt.title('Grid comparison')
plt.legend(['Initial nodes. Ritz = ' + str(round(history.history['loss'][0], 6)), 
            'Adapted nodes. Ritz = ' + str(round(history.history['loss'][-1], 6))])
plt.yticks([])
plt.grid(True)
plt.savefig('../Figures/grid_comparison' + str(nn) + '_iter' + str(iterations) + '.png')

plt.show()
