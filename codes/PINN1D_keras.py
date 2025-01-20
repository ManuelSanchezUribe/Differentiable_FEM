# -*- coding: utf-8 -*-
'''
Created on Oct 2023

@authors:   Jamie Taylor (CUNEF)
            Manuela Bastidas (UPV/EHU)
            Tomas Teijeiro (BCAM)
            https://www.mathmode.science/
            
Updated on Jan 2025 by:
    Ángel Javier Omella
'''

# This code presents a simple implementation of Physics-Informed
# Neural Networks (PINNs) as a collocation method. -- <50 lines of PINNS--

# In this 1D example, we utilize Keras for constructing neural networks
# and JAX in the backend.


import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import os

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
        return  loss


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
nn = 32
# Number of hidden layers
nl = 4
# Number of output neurons
n_output = 32
# Number of training iterations
iterations = 5000

# Initialize the neural network model for the approximate solution
model = make_special_model(nn)

init_nodes = model(jnp.array([1]))

# Big model including the  loss
loss_model = make_loss_model(model)

# Optimizer (Adam optimizer with a specific learning rate)
optimizer = keras.optimizers.Adam(learning_rate=1e-2)

# Compile the loss model with a custom loss function (tricky_loss)
loss_model.compile(optimizer=optimizer, loss=tricky_loss)

# Train the model using a single training data point ([1.], [1.]) for a
# specified number of epochs (iterations)
history = loss_model.fit(jnp.array([1.]), jnp.array([1.]), epochs=iterations)


#Plot loss history
plt.figure()
plt.plot(history.history['loss'])
plt.savefig('loss.png')

# # Define the problem domain and mesh
# n_elements = 10  # Number of elements
# n_nodes = n_elements + 1

# # Run the solver
# theta = jax.random.uniform(key=jax.random.PRNGKey(10),shape=(1,n_nodes))

# # node_coords, u = solve(theta)

node_coords, u = solve(model(jnp.array([1])))
init_coords, o = solve(init_nodes)


# # Output results
print("Node coordinates:", node_coords)
# print("Solution u:", u)
# print(val)

import matplotlib.pyplot as plt
from matplotlib import rcParams


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
plt.figure()
plt.plot(node_coords, u,'o--', color='b')
plt.plot(node_coords, np.zeros(len(node_coords)),'o', color='r')
plt.plot(init_coords, np.zeros(len(node_coords)),'o', color='k')
plt.legend(['u', 'nodes', 'initial nodes'])

plt.grid(which = 'both', axis = 'both', linestyle = ':', color = 'gray')
plt.tight_layout()

plt.savefig('plot.png')
plt.show()