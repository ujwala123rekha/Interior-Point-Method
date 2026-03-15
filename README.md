Interior-Point Method for Constrained Convex Optimization
Overview

This project demonstrates the implementation and comparison of optimization techniques for solving a convex optimization problem with linear constraints. The goal is to analyze how different optimization algorithms behave when constraints are present.

Three optimization methods are implemented and compared:

Gradient Descent (GD)

Projected Gradient Descent (PGD)

Interior-Point Method (IPM)

The project visualizes convergence behavior and optimization paths to better understand the effectiveness of each algorithm.

Problem Formulation

The optimization problem is defined as:

Minimize:

f(x) = x1² + x2²

Subject to the constraint:

x1 + x2 ≤ 1

This is a convex optimization problem, where the objective function is quadratic and the feasible region is defined by a linear inequality.

Optimization Methods Implemented
1. Gradient Descent

A basic optimization algorithm that iteratively updates parameters using the negative gradient of the objective function.

Update rule:

x(t+1) = x(t) - α ∇f(x)

Limitation: Does not enforce constraints.

2. Projected Gradient Descent

An extension of gradient descent that ensures solutions remain within the feasible region by projecting infeasible solutions back onto the constraint boundary.

3. Interior-Point Method

Interior-point methods solve constrained optimization problems by introducing a barrier function that prevents the solution from crossing constraint boundaries.

Barrier formulation:

F(x) = f(x) - μ log(1 - (x1 + x2))

This keeps the solution inside the feasible region while optimizing the objective function.

Technologies Used

Python

NumPy

Matplotlib

Project Structure
project/
│
├── optimization.py
├── README.md
├── report.pdf
└── results/
    ├── convergence_plot.png
    └── optimization_path.png
Installation

Install the required Python libraries:

pip install numpy matplotlib
Running the Project

Run the Python script:

python optimization.py

The script will:

Compute optimization using GD, PGD, and IPM

Display convergence graphs

Plot optimization paths

Results

The experiments show that:

Gradient Descent converges quickly but may violate constraints.

Projected Gradient Descent maintains feasibility but may oscillate near boundaries.

Interior-Point Method ensures smooth convergence while respecting constraints.

These results demonstrate the effectiveness of interior-point methods for constrained convex optimization problems.

Visualization

The project generates two main plots:

Optimizer Convergence Comparison

Optimization Paths for Each Method

These graphs help visualize how each algorithm approaches the optimal solution.

Conclusion

This project highlights the importance of handling constraints in optimization problems. Interior-point methods provide a powerful approach for maintaining feasibility while achieving stable convergence.

Such techniques are widely used in machine learning, operations research, and engineering optimization problems.
