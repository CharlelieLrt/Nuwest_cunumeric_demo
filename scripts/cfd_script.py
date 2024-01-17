# %% [markdown]
# # CUnumeric-based Navier-Stokes solver
# 
# In this hands-on demo, we will implement a simple Navier Stokes
# finite-difference solver based on CUnumeric. Since CUnumeric is a drop-in
# replacement for Numpy, this is (supposed to be) very easy! We will also couple
# our CFD simulations with some simple Machine Learning tasks (e.g. PCA for data
# compression). We will also discuss the performance of our application using
# tools provided by Legate.
# 
# 1. [Introduction](#introduction) (What are Cunumeric and Legate? + useful resources)
# 2. [Getting started](#getting-started) (How to install and run Cunumeric)
# 3. [A simple Navier-Stokes solver](#a-simple-navier-stokes-solver)
# 4. [Performance](#performance-evaluation) (evaluation, profiling)
# 5. [A better implementation](#a-better-implementation) (?)
# 6. [Add some Machine Learning](#add-some-machine-learning)
# 7. [Use of heterogeneous resources](#use-heterogeneous-resources) (how to dispatch computations to GPUs/CPUs)
# 
# For any question, feel free to contact me at [cl2292@stanford.edu](mailto:cl2292@stanford.edu), or directly
# reach out to the [Legate](https://github.com/nv-legate/legate.core) or
# [CUnumeric](https://github.com/nv-legate/cunumeric) developers. If you need
# help (e.g. API coverage, best practice, etc...), the most useful resource is the [Cunumeric documentation](https://nv-legate.github.io/cunumeric/23.11/).

# %% [markdown]
# ## Introduction
# 
# See [slides](../slides/Cunumeric_slides_demo.pptx).

# %% [markdown]
# ## Getting started
# 
# Note: Since CUnumeric is a drop-in replacement for Numpy, you can first
# write your application in vanilla Numpy, and then install CUnumeric later.
# 
# First case (the easy one), if you have a linux-64 system, and have either an
# Nvidia GPU Volta or later (or if you do not intend to use GPU support), then
# Cunumeric and Legate are available as a Conda package:
# ```console
# conda install -c nvidia -c conda-forge -c legate cunumeric
# ```
# 
# Otherwise, you need to first build Legate from source, based on [this
# guide](https://github.com/nv-legate/legate.core/blob/branch-24.01/BUILD.md). It
# is strongly advised to build Legate in a conda environment.
# ```console
# git clone https://github.com/nv-legate/legate.core.git
# ```
# Legate provides a script to help you set-up a conda environment (replace the
# python version and OS with appropriate values, run the script with `--help` to
# see all available options):
# ```console
# ./scripts/generate-conda-envs.py --python 3.10 --os osx
# ```
# After installing the dependencies, Legate can be built with:
# ```console
# ./install.py --max-dim 4 --openmp --clean
# ```
# 
# After completing the installation of Legate, you can clone and build Cunumeric:
# 
# ```console
# git clone https://github.com/nv-legate/cunumeric.git
# ```
# 
# then:
# 
# ```console
# ./install.py --max-dim 4 --openmp --clean
# ```
# 
# __Running Legate__
# 
# There are several ways to run Legate:
# 1. In a Jupyter notebook (limited to single node execution)
# 2. Using the standard Python interpreter (limited to single node execution)
# 3. Using the `legate` driver (for multi-node execution and more fine-grained control)
# 
# This noetbook can be executed locally, or on a supercomputer, following [these instructions](https://nv-legate.github.io/legate.core/README.html#running-with-jupyter-notebook).

# %% [markdown]
# ## A simple Navier-Stokes solver
# 
# We will implement a solver that solver integrates the three-dimensional, compressible, single-component Navier-Stokes equations
# for an ideal gas. These equations and numerical integration are detailed below.
# 
# <img src="../images/NS_equation.png" width="700">
# 

# %%
# Import Cunumeric instead of numpy
import cunumeric as num
from math import pi
import numpy as np
import matplotlib.pyplot as plt

# Define some constants
constants = {
    "grid": (64,) * 3,
    "grid_ghost": (64 + 2,) * 3,
    "length": 1.0,
    "rho_0": 1.0,
    "p_0": 1e5,
    "T_0": 300.0,
    "gamma": 1.4,
    "r_gas": 8.3144598 / 32e-3,
    "mu": 2.0e-5,
    "lambda": 0.026,
    "forcing_amplitude": 0.5,
    "dt": 1e-5
}

# %% [markdown]
# ### Convection fluxes
# 
# First let's implement a function that computes the convection fluxes for the 5
# flow variables.
# 
# <img src="../images/Convection_fluxes.png" width="700">

# %%
def center_to_face(dir, x):
    if dir == 0:
        idx_m = (slice(0, -1), slice(None), slice(None))
        idx_p = (slice(1, None), slice(None), slice(None))
    elif dir == 1:
        idx_m = (slice(None), slice(0, -1), slice(None))
        idx_p = (slice(None), slice(1, None), slice(None))
    elif dir == 2:
        idx_m = (slice(None), slice(None), slice(0, -1))
        idx_p = (slice(None), slice(None), slice(1, None))
    return 0.5 * (x[(..., *idx_m)] + x[(..., *idx_p)])


def plus(dir, delta=1):
    if dir == 0:
        return (slice(delta, None), slice(None), slice(None))
    elif dir == 1:
        return (slice(None), slice(delta, None), slice(None))
    elif dir == 2:
        return (slice(None), slice(None), slice(delta, None))


def minus(dir, delta=1):
    if dir == 0:
        return (slice(None, -delta), slice(None), slice(None))
    elif dir == 1:
        return (slice(None), slice(None, -delta), slice(None))
    elif dir == 2:
        return (slice(None), slice(None), slice(None, -delta))


def get_interior(exclude=None):
    if isinstance(exclude, int) or exclude is None:
        exclude = (exclude,)
    slices = ()
    for i in range(3):
        if i not in exclude:
            slices += (slice(1, -1),)
        else:
            slices += (slice(None),)
    return slices


def convection(rho, rhou, rhoE):
    for dir in range(3):
        interior = get_interior(exclude=dir)
        inv_dx = 1.0 / (constants["length"] / constants["grid"][dir])
        F_conv = num.zeros(
            (5,) + tuple(d-1 if i == dir else d-2 for i, d in enumerate(rho.shape))
        )
        # rho flux
        F_conv[0, ...] = center_to_face(dir, rhou[(dir, *interior)])
        # Momentum flux
        u = rhou[dir, ...] / rho
        p = (constants["gamma"] - 1.0) * (
            rhoE - 0.5 * num.sum(rhou * rhou, axis=0) / rho
        ) + rho * constants["T_0"] * constants["gamma"] * constants["r_gas"]
        p_s = center_to_face(dir, p[interior])
        u_s = center_to_face(dir, u[interior])
        F_conv[1:4, ...] = center_to_face(dir, rhou[(..., *interior)]) * u_s
        F_conv[dir+1, ...] += p_s
        # Energy flux
        F_conv[4, ...] = (center_to_face(dir, rhoE[interior]) + p_s) * u_s
        dwdt = (F_conv[(..., *plus(dir))] - F_conv[(..., *minus(dir))]) * inv_dx
        return dwdt


# %% [markdown]
# ### Diffusion fluxes
# 

# %%
def compute_stress_tensor(rho, rhou, dir):
    # Create some slices for indices i-1 and i+1
    interior = get_interior(exclude=dir)
    inv_dx = 1.0 / (constants["length"] / constants["grid"][dir])
    if dir == 0:
        transverse = (1, 2)
    elif dir == 1:
        transverse = (2, 0)
    elif dir == 2:
        transverse = (0, 1)
    u = rhou / rho
    # Start with [2*du/dx_s, dv/dx_s, dz/dx_s]
    tau = (
        u[(..., *interior)][(..., *plus(dir))] -
        u[(..., *interior)][(..., *minus(dir))]
    ) * inv_dx
    tau[dir, ...] *= 2
    # Add transverse derivatives: [-dv/dy_s, du/dy_s, 0]
    for dirT in transverse:
        interior_dirT = get_interior(exclude=(dir, dirT))
        inv_dy = 1.0 / (constants["length"] / constants["grid"][dirT])
        m2 = minus(dirT, delta=2)
        p2 = plus(dirT, delta=2)
        dvdy = 0.5 * (u[(dirT, *p2)] - u[(dirT, *m2)]) * inv_dy
        dudy = 0.5 * (u[(dirT, *p2)] - u[(dirT, *m2)]) * inv_dy
        interior_dirT = get_interior((dir, dirT))
        dvdy_s = center_to_face(dir, dvdy[interior_dirT])
        dudy_s = center_to_face(dir, dudy[interior_dirT])
        tau[dir, ...] -= dvdy_s
        tau[dirT, ...] += dudy_s
    tau[dir, ...] *= constants["mu"] * 2./3
    return tau

def diffusion(rho, rhou, rhoE, dwdt):
    for dir in range(3):
        interior = get_interior(exclude=dir)
        inv_dx = 1.0 / (constants["length"] / constants["grid"][dir])
        F_diff = num.zeros(
            (5,) + tuple(d-1 if i == dir else d-2 for i, d in enumerate(rho.shape))
        )
        # rho flux: zero
        # Momentum flux
        tau = compute_stress_tensor(rho, rhou, dir)
        F_diff[1:4, ...] = tau
        # Energy flux
        p = (constants["gamma"] - 1.0) * (
            rhoE - 0.5 * num.sum(rhou * rhou, axis=0) / rho
        ) + rho * constants["T_0"] * constants["gamma"] * constants["r_gas"]
        T = p / (rho * constants["r_gas"])
        dTdx = (T[interior][plus(dir)] - T[interior][minus(dir)]) * inv_dx
        u = rhou / rho
        u_s = center_to_face(dir, u[(..., *interior)])
        F_diff[4, ...] = constants["lambda"] * dTdx + num.einsum(
            "i...,i...->...", tau, u_s)
        dwdt = (F_diff[(..., *plus(dir))] - F_diff[(..., *minus(dir))]) * inv_dx
        return dwdt

# %% [markdown]
# ### Initial conditions and forcing

# %%
def initialization(rho, rhou, rhoE, x, y, z):
    u = num.zeros((3,) + rho.shape)
    u[:] += (num.sin(2 * pi * x) * num.sin(2 * pi * y ) * num.sin(2 * pi * z ))
    u[0, ...] += 0.5 * (1 - y) * y + 0.5 * (1 - z) * z
    u[1, ...] += 0.5 * (1 - x) * x + 0.5 * (1 - z) * z
    u[2, ...] += 0.5 * (1 - x) * x + 0.5 * (1 - y) * y
    rhou[:] = rho * u
    rhoE[:] = 0.5 * rho * num.sum(u, axis=0) - \
        (rho * constants["T_0"] * constants["gamma"] * constants["r_gas"] -
         - constants["p_0"]/(constants["gamma"] - 1))

def forcing(rhou):
    interior = get_interior(exclude=None)
    drhoudt = constants["forcing_amplitude"] * rhou[(..., *interior)]
    return drhoudt    

# %% [markdown]
# ### Boundary conditions

# %%
def boundary_conditions(rho, rhou, rhoE):
    # Z direction (left side)
    rho[:, :, 0] = rho[:, :, -2]
    rhou[:, :, :, 0] = rhou[:, :, :, -2]
    rhoE[:, :, 0] = rhoE[:, :, -2]
    # Z direction (right side)
    rho[:, :, -1] = rho[:, :, 1]
    rhou[:, :, :, -1] = rhou[:, :, :, 1]
    rhoE[:, :, -1] = rhoE[:, :, 1]
    # Y direction (left)
    rho[:, 0, :] = rho[:, -2, :]
    rhou[:, :, 0, :] = rhou[:, :, -2, :]
    rhoE[:, 0, :] = rhoE[:, -2, :]
    # Y direction (right)
    rho[:, -1, :] = rho[:, 1, :]
    rhou[:, :, -1, :] = rhou[:, 1, :]
    rhoE[:, -1, :] = rhoE[:, 1, :]
    # X direction (left)
    rho[0, :, :] = rho[-2, :, :]
    rhou[:, 0, :, :] = rhou[:, -2, :, :]
    rhoE[0, :, :] = rhoE[-2, :, :]
    # X direction (right)
    rho[-1, :, :] = rho[1, :, :]
    rhou[:, -1, :, :] = rhou[:, 1, :, :]
    rhoE[-1, :, :] = rhoE[1, :, :]

# %% [markdown]
# ### Wrap it up

# %%
def run_simulation(niter):

    # Create variables
    dx = constants["length"] / constants["grid"][0]
    rho = num.full(constants["grid_ghost"], constants["rho_0"])
    rhou = num.zeros((3,) + constants["grid_ghost"])
    rhoE = num.zeros(constants["grid_ghost"])
    dwdt = num.zeros((5,) + constants["grid"])
    x = num.linspace(
            start=-dx,
            stop=constants["length"] + dx,
            num=constants["grid"][0] + 2
    )[:, None, None]
    y = num.linspace(
            start=-dx,
            stop=constants["length"] + dx,
            num=constants["grid"][1] + 2
    )[None, :, None]
    z = num.linspace(
            start=-dx,
            stop=constants["length"] + dx,
            num=constants["grid"][2] + 2
    )[None, None, :]
    
    # Initial solution
    initialization(rho, rhou, rhoE, x, y, z)
    
    interior = get_interior(exclude=None)
    
    # Time loop
    for i in range(niter):
        if i % 100 == 0:
                print(f"Iteration: {i}")
        dwdt[:] = 0.
        # Set boundary values in ghost cells
        boundary_conditions(rho, rhou, rhoE)
        # Compute fluxes and forcing
        dwdt[:] += convection(rho, rhou, rhoE)
        dwdt[1:4, ...] += forcing(rhou)
        dwdt[:] += diffusion(rho, rhou, rhoE, dwdt)
        # Update variables
        rho[interior] += constants["dt"] * dwdt[0, ...]
        rhou[(..., *interior)] += constants["dt"] * dwdt[1:4, ...]
        rhoE[interior] += constants["dt"] * dwdt[4, ...]
    return rho, rhou, rhoE
    

# %%
if __init__ == "__main__":
    rho, rhu, rhoE = run_simulation(1000)

# %% [markdown]
# ## Performance evaluation

# %%
# init() # Initialization step

# # Do few warm-up iterations
# for i in range(n_warmup_iters):
#     compute()

# start = time()
# for i in range(niters):
#     compute()
# end = time()

# %% [markdown]
# ## A better implementation

# %% [markdown]
# ## Add some Machine Learning

# %% [markdown]
# ## Use heterogeneous resources


