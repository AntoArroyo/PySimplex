# Simplex Solver Project

This project implements the Simplex method for solving linear programming problems. The method is applied to maximize an objective function subject to a set of linear constraints. The project reads problem data from an Excel file and performs the optimization step by step, displaying intermediate steps and the final solution.

## Features

- Solves linear programming problems using the Simplex method.
- Reads input data (objective function and constraints) from an Excel file (can read .csv files using -csv).
- Displays each step of the Simplex algorithm, including:
  - Tableau updates.
  - Pivot operations.
  - Intermediate solutions.
- Identifies and handles cases such as unbounded or infinite solutions.
- Outputs the optimal solution and the values of the variables.
- For cases with 2 variables shows the graphical solution.

---

## Getting Started

### Prerequisites

- Python 3.x
- Required libraries:
  - `pandas`
  - `numpy`
  - `matplotlib`

You can install the dependencies using:

```bash
pip install pandas numpy matplotlib
