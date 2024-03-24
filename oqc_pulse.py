import pennylane as qml
import numpy as np
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Setting up JAX for 64-bit precision
jax.config.update("jax_enable_x64", True)

# Define Pauli matrices
X, Y, Z = qml.PauliX(0), qml.PauliY(0), qml.PauliZ(0)

# Qubit frequency
omega = 2 * jnp.pi * 5.

# Amplitude function for the Hamiltonian
def amp(nu):
    def wrapped(p, t):
        return jnp.pi * jnp.sin(nu*t + p)
    return wrapped

# Hamiltonian definition
H = -omega/2 * qml.PauliZ(0)
H += amp(omega) * qml.PauliY(0)

# QNode for evolving the qubit state
@jax.jit
@qml.qnode(qml.device("default.qubit", wires=1), interface="jax")
def trajectory(params, t):
    qml.evolve(H)((params,), t, return_intermediate=True)
    return [qml.expval(op) for op in [X, Y, Z]]

# Compute time series for 10000 samples for two phases
ts = jnp.linspace(0., 1., 10000)
res0 = trajectory(0., ts)
res1 = trajectory(jnp.pi/2, ts)

# Plot the evolution in the Bloch sphere
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot(*res0, "-", label="$\\phi=0$")
ax.plot(*res1, "-", label="$\\phi=\\pi/2$")
ax.legend()
plt.show()
