########################################################################################
# Workshop: Coding for PDEs with Neural Networks
# Date: 2025-24-01
# Author: Danilo Aballay, Vicente Iligaray, Ignacio Tapia y Manuel Sánchez
########################################################################################

import sympy as sp

# Define the variable and parameter
x = sp.Symbol('x')
alpha = 0.6
# alpha = sp.Symbol('alpha', positive=True)

# Define the function u(x)
u = x**alpha
# up = alpha*x**(alpha-1)
up = sp.diff(u, x)
# f = -alpha*(alpha-1)*x**(alpha-2)
f = -sp.diff(up, x)

# Compute the definite integral from 0 to 1
integral = sp.integrate(0.5*up*up - f*u, (x, 0, 1))# - alpha

# Display the result
sp.pprint(integral)
