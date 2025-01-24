import jax.numpy as np
from jax import random, grad, jit, vmap
import jax
import jax.nn as nn
from jax.scipy.optimize import minimize
from jax import lax
import numpy as onp
from jax.experimental.ode import odeint
from jax.example_libraries import optimizers
from jax import block_until_ready
from functools import partial

from Laplace_JAXDense import solve_and_loss

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

@jit
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
    ritz   = jax.vmap(solve_and_loss)(thetas)[:, None]
    return np.sum(ritz)

@partial(jit, static_argnums=(0,))
def step(loss, i, opt_state, sigmas):
    params = get_params(opt_state)
    grads  = grad(loss)(params, sigmas)
    return opt_update(0, grads, opt_state)

def train(loss_fn, opt_state,sigmas, nIter=10000):
    train_loss = []
    for it in range(nIter):
        params = get_params(opt_state)
        grads  = grad(loss_fn)(params, sigmas)
        opt_state = opt_update(0, grads, opt_state)
        # opt_state = step(loss, it, opt_state, A, C, t_span, lb, ub)

        if it % 100 == 0:
            params = get_params(opt_state)
            loss_val = loss_fn(params, sigmas)
            train_loss.append(loss_val)
            if it % 500 == 0:
              print(f"Iteración {it}, pérdida: {loss_val:.4e}")

    return get_params(opt_state), train_loss


sigmas = np.linspace(0.1, 10, 100)
sigmas = sigmas[:, None]

# Parámetros de la red neuronal
layers = [1, 10, 10, 16]

# Entrenamiento
opt_init, opt_update, get_params = optimizers.adam(1e-4)
params    = init_params(layers, key)
opt_state = opt_init(params)
Iter      = 1000

# Configurar los parámetros
trained_params_2, trained_error_2 = train(loss, opt_state, sigmas, Iter)
