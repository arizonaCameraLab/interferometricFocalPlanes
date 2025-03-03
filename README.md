# Interferometric Focal Planes

This repository contains the supporting code for the article **"Interferometric Focal Planes" (under review)**. The code implements Monte Carlo simulations to estimate parameters in several prototype objects, including:

- Two point sources  
- A line source  
- An edge source  
- A right-angle source  

These simulations explore **fixed and adaptive measurement strategies** for mutual intensity measurement and compare them with the estimation results from intensity measurements. **Jacobian matrices** of the forward models are also calculated to determine the degrees of freedom of the system.

## ⚙️ **Requirements**

This code is written in **MATLAB R2022b** and requires the following toolboxes:

- **Parallel Computing Toolbox** (for `parfor`, used in Monte Carlo simulations)  
- **Optimization Toolbox** (for `fminunc`, used in Maximum Likelihood Estimation)  

If running the code without a **Parallel Computing Toolbox**, replace `parfor` loops with `for` loops to ensure compatibility.
