# test_weyl_jacobians.py
import pytest
import numpy as np
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)
import jax.scipy.linalg as jsl

# ------------------------------
# Pauli and helpers
# ------------------------------
I = jnp.eye(2, dtype=jnp.complex128)
X = jnp.array([[0, 1], [1, 0]], dtype=jnp.complex128)
Y = jnp.array([[0, -1j], [1j, 0]], dtype=jnp.complex128)
Z = jnp.array([[1, 0], [0, -1]], dtype=jnp.complex128)

def kron(A, B):
    return jnp.kron(A, B)

XX = kron(X, X)
YY = kron(Y, Y)
ZZ = kron(Z, Z)

def Rz(theta):
    return jsl.expm(-0.5j * theta * Z)

def Ry(phi):
    return jsl.expm(-0.5j * phi * Y)

def K(theta, psi, phi):
    return Rz(theta) @ Ry(psi) @ Rz(phi)

def dK_dtheta(theta, psi, phi):
    return (-0.5j) * Z @ K(theta, psi, phi)

def dK_dpsi(theta, psi, phi):
    return Rz(theta) @ ((-0.5j) * Y @ Ry(psi)) @ Rz(phi)

def dK_dphi(theta, psi, phi):
    return Rz(theta) @ Ry(psi) @ ((-0.5j) * Z @ Rz(phi))

def V(a, b, c):
    H = a * XX + b * YY + c * ZZ
    return jsl.expm(1j * H)

def dV_da(a, b, c):
    # XX, YY, ZZ commute, so d/da exp(i(aXX + bYY + cZZ)) = i XX V
    return 1j * XX @ V(a, b, c)

def G(params):
    # params = (t1,p1,f1, t2,p2,f2, a,b,c, t3,p3,f3, t4,p4,f4)
    t1,p1,f1, t2,p2,f2, a,b,c, t3,p3,f3, t4,p4,f4 = params
    K1 = K(t1,p1,f1)
    K2 = K(t2,p2,f2)
    K3 = K(t3,p3,f3)
    K4 = K(t4,p4,f4)
    A  = kron(K1, K2)
    B  = kron(K3, K4)
    return A @ V(a,b,c) @ B

def dG_da(params):
    t1,p1,f1, t2,p2,f2, a,b,c, t3,p3,f3, t4,p4,f4 = params
    K1 = K(t1,p1,f1)
    K2 = K(t2,p2,f2)
    K3 = K(t3,p3,f3)
    K4 = K(t4,p4,f4)
    A  = kron(K1, K2)
    B  = kron(K3, K4)
    return A @ (1j * XX @ V(a,b,c)) @ B

def central_diff_5pt(f, x, h=None):
    if h is None:
        # scale h to the magnitude of x; avoid under/overflow
        scale = max(1.0, abs(x))
        h = (np.finfo(float).eps ** 0.25) * scale   # ~1e-4 for O(h^4)
    return (-f(x+2*h) + 8*f(x+h) - 8*f(x-h) + f(x-2*h)) / (12*h)

def frob_norm(A):
    return jnp.sqrt(jnp.vdot(A.reshape(-1), A.reshape(-1)).real)

# ------------------------------
# Tests: single-qubit Jacobians
# ------------------------------
@pytest.mark.parametrize("theta,psi,phi", [
    (0.7, -0.9, 1.3),
    (-1.2, 0.4, -0.2),
])
def test_dK_dtheta(theta, psi, phi, tol=1e-9):
    f = lambda th: K(th, psi, phi)
    num = central_diff_5pt(f, theta)
    ana = dK_dtheta(theta, psi, phi)
    err = frob_norm(num - ana)
    base = 1.0 + frob_norm(num) + frob_norm(ana)
    assert err / base < tol

@pytest.mark.parametrize("theta,psi,phi", [
    (0.25, 0.8, -0.6),
    (1.0, -1.1, 0.3),
])
def test_dK_dpsi(theta, psi, phi, tol=1e-9):
    f = lambda ps: K(theta, ps, phi)
    num = central_diff_5pt(f, psi)
    ana = dK_dpsi(theta, psi, phi)
    err = frob_norm(num - ana)
    base = 1.0 + frob_norm(num) + frob_norm(ana)
    assert err / base < tol

@pytest.mark.parametrize("theta,psi,phi", [
    (-0.75, 0.2, 0.9),
    (0.33, -0.7, -1.2),
])
def test_dK_dphi(theta, psi, phi, tol=1e-9):
    f = lambda ph: K(theta, psi, ph)
    num = central_diff_5pt(f, phi)
    ana = dK_dphi(theta, psi, phi)
    err = frob_norm(num - ana)
    base = 1.0 + frob_norm(num) + frob_norm(ana)
    assert err / base < tol

# ------------------------------
# Test: two-qubit nonlocal derivative dG/da
# ------------------------------
@pytest.mark.parametrize("params", [
    # (t1,p1,f1, t2,p2,f2, a,b,c, t3,p3,f3, t4,p4,f4)
    (0.1, -0.3, 0.7,   0.2, 0.9, -0.5,   0.4, -0.2, 0.6,   -0.1, 0.3, 0.2,   -0.4, -0.6, 0.8),
    (1.2, 0.1, -0.8,   -0.7, 0.5, 0.9,   -0.3, 0.2, -0.4,   0.6, -0.9, 0.1,   0.5, 0.7, -1.1),
])
def test_dG_da(params, tol=5e-9):
    # numeric derivative w.r.t. 'a'
    idx_a = 6  # position of 'a' in the tuple
    def f_of_a(a_val):
        p = list(params)
        p[idx_a] = a_val
        return G(tuple(p))

    a0 = params[idx_a]
    num = central_diff_5pt(f_of_a, a0)
    ana = dG_da(params)
    err = frob_norm(num - ana)
    base = 1.0 + frob_norm(num) + frob_norm(ana)
    assert err / base < tol
