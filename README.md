# Gaussian-Process-Regression
This project finds its greatest motivation in the field of geoscience, where approximating unknown properties of the subsurface by means of Gaussian processes is a standard procedure. In particular, we know that scientists have at their disposal limited data collected through boreholes. The goal of the project is therefore to derive information about the entire permeability field of the area being studied by means of Gaussian Process Regression, that is, to make use of conditional Gaussian processes in order to reliably predict the unknown permeability field.
The use of Gaussian Process Regression in this project has allowed us to successfully reconstruct a onedimensional function (Task1) and estimate the permeability field through two-dimensional observations in a geoscience problem (Task2).

### Team
- Francesco Pettenon: francesco.pettenon@epfl.ch
- Francesca Venturi: francesca.venturi@epfl.ch

## Simulations
- `GPR1.py`: simulation of task1: Recovering a simple function
- `GPR2.py`: simulation of task2: GP Regression on a permeability field

## Helpers
- `kernels.py`
- `domains.py`
- `optimizer.py`
- `processes.py`
- `MonteCarlo.py`
- `plots.py`

## Data
The reference field given as a 110 × 60 `numpy array` object, can be found under the name `true perm.npy`

## Documents
- `Gaussian_process_regression.pdf`: the text of the project
- `Stochastic_Simulation_Report.pdf`: the text of the report
- `Stochastic_Simulation_Presentation.pdf`: the presentation of the project 
