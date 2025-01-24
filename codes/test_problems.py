########################################################################################
# Workshop: Coding for PDEs with Neural Networks
# Date: 2025-24-01
# Author: Danilo Aballay, Vicente Iligaray, Ignacio Tapia y Manuel Sánchez
########################################################################################

import jax
import jax.numpy as jnp
from jax import jit
# 1d second order elliptic problem-Dirichlet boundary condition
# -(u'(x)*c(X))' + a(x) = f(x)
# u(0) = g(0), u(1) = g(1)

class elliptic1d:
    def __init__(self, f, g0, g1, c, a, u=None):
        self.x0 = 0.0
        self.x1 = 1.0
        self.f = f
        self.g0 = g0
        self.g1 = g1
        self.c = c
        self.a = a
        self.u = u
    
def problem(problem_number):
    problems_data = [
        {
            "f": lambda x: 0*x,
            "g0": 0.5,  # Left Dirichlet boundary condition,
            "g1": -0.5,  # Left Dirichlet boundary condition
            "a": lambda x: 1,  # Constant coefficient a(x)
            "c": lambda x: 0*x,  # Constant coefficient c(x)
        },
        {
            "f": lambda x: jnp.sin(jnp.pi*x),
            "g0": 0,  # Left Dirichlet boundary condition,
            "g1": 0,  # Left Dirichlet boundary condition
            "a": lambda x: 1,  # Constant coefficient a(x)
            "c": lambda x: 0*x,  # Constant coefficient c(x)
        }
    ]
    return elliptic1d(problems_data[problem_number])

