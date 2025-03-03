# **Official Repository for *Interferometric Focal Planes***

This repository contains the supporting code for the article ***Interferometric Focal Planes* (under review)**. The code implements Monte Carlo simulations to estimate parameters in several prototype objects, including:

- Two point sources
- A line source
- An edge source
- A right-angle source

These simulations explore fixed and adaptive measurement strategies for mutual intensity measurement and compare them with the estimation results from intensity measurements. Jacobian matrices of the forward models are also calculated to determine the degrees of freedom of the system.

---

## 📌 **Code Structure**

SimulationCode/  
│── Corner/  
│   ├── CornerEstimation.m  
│   ├── JacobianCorner.m  
│   ├── MeasureCorner.m  
│  
│── Edge/  
│   ├── EdgeEstimation.m  
│   ├── JacobianEdge.m  
│   ├── MeasureEdge.m  
│  
│── Line/  
│   ├── Jacobian1Line.m  
│   ├── LineEstimation.m  
│   ├── MeasureLine.m  
│  
│── twopoints/  
│   ├── Jacobian2P.m  
│   ├── Measurel2P2D.m  
│   ├── two_pts_estimation.m  
│  
├── README.md  

The code is organized into different folders, each corresponding to a prototype object.

Each folder contains three MATLAB scripts:

1. **Main Program** (`*_Estimation.m`)  
   - Runs Monte Carlo simulations for parameter estimation.  
   - Implements fixed and adaptive measurement strategies.  
   - Uses a forward model to simulate measurements.  
   - Includes visualization code to generate comparison plots.  

2. **Forward Model** (`Measure*.m`)  
   - Defines the physical measurement process for the given object.  
   - Used in the Monte Carlo simulations to generate synthetic data.  

3. **Jacobian Calculation** (`Jacobian*.m`)  
   - Computes the Jacobian matrix of the forward model.  

   - Not used in Monte Carlo simulations, but provides theoretical insights.  

   - Used for visualizing degrees of freedom in parameter estimation. 

---

## 🚀 **How to Use**

1. **Clone** or **download** this repository.

2. Open **MATLAB** in the `SimulationCode` directory.

3. Choose a geometry and run its **main script**. For example, for the corner (right-angle) source:

   ```matlab
   run Corner/CornerEstimation.m
   ```

## 📝 **Citation & License**

If you use this code, please cite our article *Interferometric Focal Planes* (once published).

**License**: This code is provided for research purposes only.

