import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import os

os.environ["KERAS_BACKEND"] = "jax"

import keras

from Laplace_JAXDense import solve_and_loss, solve
from PINN1D_keras import make_special_model, tricky_loss, make_loss_model
# Set the random seed
np.random.seed(1234)
keras.utils.set_random_seed(1234)

dtype='float64' # double precision set to default in the SCR functions
jax.config.update("jax_enable_x64", True)
keras.backend.set_floatx(dtype)


######################################################
################ Save results ########################
######################################################

orders = [2,3,4,5,6,7,8]
iterations = 10000


save_coords = []
for i in orders:
    nn = 2**i

    # Initialize the neural network model for the approximate solution
    model = make_special_model(nn)

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


    # Coordenates
    node_coords, u = solve(model(jnp.array([1])))

    # Save node_coords
    save_coords.append(node_coords)

# Save list in csv file
import csv
with open('save_coords.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    for coords in save_coords:
        writer.writerow(coords)