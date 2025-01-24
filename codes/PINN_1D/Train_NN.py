import os
import jax
import jax.numpy as jnp
import numpy as np
import keras
from NN_model import make_model, make_loss_model, tricky_loss
from Laplace_JAXDense import solve_and_loss, solve
os.environ["KERAS_BACKEND"] = "jax"


# Set the random seed
np.random.seed(1234)
keras.utils.set_random_seed(1234)

dtype='float64' # double precision set to default in the SCR functions
jax.config.update("jax_enable_x64", True)
keras.backend.set_floatx(dtype)

###############################################################################
#                  Training the model                                        #
###############################################################################

neurons    = 10
n_layers   = 3
n_nodes    = int(2**4)
iterations = 1000
sigmas     = jnp.array([[0.5], [1], [2], [4]], dtype = dtype)

# Create the model
model = make_model(neurons, n_layers, n_nodes)

# Create loss model
loss_model = make_loss_model(model, solve_and_loss, sigmas)

# Compile the model
# Optimizer (Adam optimizer with a specific learning rate)
optimizer = keras.optimizers.Adam(learning_rate=10**-3)

# Compile the loss model with a custom loss function (tricky_loss)
loss_model.compile(optimizer=optimizer, loss=tricky_loss)

# Train the model using a single training data point ([1.], [1.]) for a
# specified number of epochs (iterations)

# Adatative learning rate
def lr_schedule(epoch, lr):
    if epoch >= 5000:
        return 1e-3
    return lr
lr_scheduler = keras.callbacks.LearningRateScheduler(lr_schedule)

history = loss_model.fit(sigmas, sigmas, epochs=iterations)
