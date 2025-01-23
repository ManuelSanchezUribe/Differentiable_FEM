import jax
import jax.numpy as jnp
import numpy as np
import keras

from Laplace_JAXDense import solve_and_loss, solve

# Set the random seed
np.random.seed(1234)
keras.utils.set_random_seed(1234)

dtype='float64' # double precision set to default in the SCR functions
jax.config.update("jax_enable_x64", True)
keras.backend.set_floatx(dtype)

sigmas     = jnp.array([0.5, 1, 2, 4], dtype = dtype)

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

def make_model(neurons, n_layers, n_nodes, activation = 'tanh'):

    xvals = keras.layers.Input(shape=(1,), name='x_input',dtype=dtype)

    # First layer
    l1 = keras.layers.Dense(neurons, activation=activation, dtype=dtype)(xvals)

    for l in range(n_layers-2):
    # Hidden layers
        l1 = keras.layers.Dense(neurons, activation=activation, dtype=dtype)(l1)
    # Last layer
    output = keras.layers.Dense(n_nodes, activation=None, dtype=dtype)(l1)

    u_model = keras.Model(inputs = xvals, outputs = output, name='u_model')

    return u_model

class loss(keras.layers.Layer):
    def __init__(self, model_theta,f, inputs, **kwargs):
        super(loss, self).__init__()
        self.model   = model_theta
        self.f       = f #  theta -> FEM -> Ritz
        self.sigma   = inputs

    def call(self,inputs):
        thetas = self.model(self.sigma)  # Calcular los valores del modelo
        return jnp.sum(jax.vmap(self.f)(thetas[:, None]))
    

def make_loss_model(model, f, inputs):
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
    output = loss(model, f, inputs)(xvals)
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
