########################################################################################
# Workshop: Coding for PDEs with Neural Networks
# Date: 2025-24-01
# Author: Danilo Aballay, Vicente Iligaray, Ignacio Tapia y Manuel Sánchez
########################################################################################

import jax.numpy as np
from jax import random, grad, jit, vmap
import jax
import jax.nn as nn
from jax.scipy.optimize import minimize
from jax import lax
import numpy as onp
from jax.experimental.ode import odeint
from jax.example_libraries import optimizers
import matplotlib.pyplot as plt
from jax import block_until_ready
from functools import partial

from Laplace_JAXDense import solve_and_loss, solve, softmax_nodes

key = random.PRNGKey(0)


# Estructura red neuronal

def init_params(layers, key):
    Ws = []
    bs = []
    for i in range(len(layers) - 1):
        std_glorot = np.sqrt(2 / (layers[i] + layers[i + 1]))
        key, subkey = random.split(key)
        Ws.append(random.normal(subkey, (layers[i], layers[i + 1])) * std_glorot)
        bs.append(np.zeros(layers[i + 1]))
    return [Ws, bs]

# @jit
def forward_pass(A_t, params):
    Ws, bs = params
    H = A_t
    for i in range(len(Ws) - 1):
        H = np.matmul(H, Ws[i]) + bs[i]
        H = np.tanh(H)
    alpha = np.matmul(H, Ws[-1]) + bs[-1]
    return alpha

@jit
def loss(params, sigmas):
    thetas = forward_pass(sigmas, params)
    mapped_function = jax.vmap(solve_and_loss, in_axes=(0, 0))
    ritz = mapped_function(thetas[:, None], sigmas)
    return np.sum(ritz)

@partial(jit, static_argnums=(0,))
def step(loss, i, opt_state, sigmas):
    params = get_params(opt_state)
    grads  = grad(loss)(params, sigmas)
    return opt_update(0, grads, opt_state)

def FEM_sol(params, sigma):
    thetas = forward_pass(sigma, params)
    return solve(thetas, sigma)

def train(loss_fn, opt_state,sigmas, nIter=10000):
    train_loss = []
    for it in range(nIter):
        params = get_params(opt_state)
        grads  = grad(loss_fn)(params, sigmas)
        opt_state = opt_update(0, grads, opt_state)
        # opt_state = step(loss, it, opt_state, A, C, t_span, lb, ub)

        if it % 10 == 0:
            params = get_params(opt_state)
            loss_val = loss_fn(params, sigmas)
            train_loss.append(loss_val)
            if it % 10 == 0:
              print(f"Iteración {it}, pérdida: {loss_val:.4e}")

    return get_params(opt_state), train_loss


#######################################################################
#                         TRAIN NN                                    #
#######################################################################

neurons = 10
layers  = 3
n_nodes = 16
iter    = 5000
sigmas  = np.linspace(0.1,100,100)
sigmas  = sigmas[:, None]

# Parámetros de la red neuronal
layers = [1, 10, 10, n_nodes]
key    = random.PRNGKey(42)

# Entrenamiento
opt_init, opt_update, get_params = optimizers.adam(1e-2)
params    = init_params(layers, key)
opt_state = opt_init(params)

# Configurar los parámetros
trained_param, trained_error = train(loss, opt_state, sigmas, iter)

# Resultados
plt.plot(trained_error)

node_coords, u = FEM_sol(trained_param, np.array([sigmas[4]]))

import matplotlib.pyplot as plt
# fig, ax = plt.subplots()
# # Plot the approximate solution obtained from the trained model
plt.figure(2)
plt.plot(node_coords, u,'o--', color='b')
# plt.plot(init_coords, np.zeros(len(node_coords)),'o', color='k', alpha=0.5, markersize=5)
# plt.plot(node_coords, np.zeros(len(node_coords)),'o', color='r', alpha=0.5, markersize=5)
plt.legend(['u', 'nodes', 'initial nodes'])
plt.xlabel('x')
plt.ylabel('u')
plt.grid(which = 'both', axis = 'both', linestyle = ':', color = 'gray')
plt.tight_layout()
