
import numpy as np


# --------------------------------------- Radial basis function ---------------------------------------
"""
A radial basis function (RBF) is a real-valued function φ whose value depends only on the distance 
from the origin, so that φ(x) = φ(∥x∥). The norm is usually Euclidean distance.
The Euclidean distance measures the length of the vector ∥x∥ = square(x•x).
or alternatively on the distance from some other point φ(x, c) = φ(∥x - c∥).

Gaussian radial basis function include: 
    r = ∥x - x_i∥
    using ε to indicate a shape parameter that can be used to scale the input of the radial kernel.
Gaussian: φ(r) = exp(-(εr)^2)
source: https://en.wikipedia.org/wiki/Radial_basis_function

Function approximation: y(x) = Σ w_i * φ(∥x - x_i∥)
where:
    y(x) - the approximating function y(x) is represented as a sum of N radial basis functions
    x_i  - each radial basis function associated with a different center x_i
    w_i  - each radial basis function weighted by an appropriate coefficient w_i
         - The weights w_i can be estimated using the matrix methods of linear least squares, 
           because the approximating function is linear in the weights w_i.
"""


# The basis function used to compute phi
# x_i       np.array, size = 4
# state     np.array, size = 4
# return    float value, φ(∥state - x_i∥) = exp(-(0.5*∥state - x_i∥)^2)
def calc_radial_basis_function(state, x_i):
    sub_res = state - x_i
    r = np.dot(sub_res, sub_res)

    return np.exp(-0.5 * r)


# ϕ(s,a) is used to select the best action according to the policy
# return array of size = (state_space_size + 1) * action_space_size = 10
def calc_phi(state, action, center_list, phi_matrix_size):
    phi = np.zeros((phi_matrix_size,))
    offset = (len(center_list[0]) + 1) * action       # correspond row to current action in matrix
    phi_row = [calc_radial_basis_function(state, x_i) for x_i in center_list]  # x_i is np array of size 4
    row_length = len(phi_row)

    phi[offset] = 1
    offset += 1
    phi[offset: offset + row_length] = phi_row
    return phi  # size = 10


# return q'(s,a,w) = dot( x(s,a)^T , w) : Approximation of action-value function
# return float
def calc_q_per_action(state, action, w, center_list, phi_matrix_size):
    phi = calc_phi(state, action, center_list, phi_matrix_size)  # calculate x(s,a) : linear combination of features
    return np.dot(phi, w)


