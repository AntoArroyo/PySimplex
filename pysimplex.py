import numpy as np
import pandas as pd
import argparse

def read_csv(file_path):
    df = pd.read_csv(file_path, header=None)
    # Objective Function
    c = df.iloc[0, :-1].dropna().values.astype(float)
    # Constraints
    A = df.iloc[1:, :-2].dropna(axis=1).values.astype(float)
    signs = df.iloc[1:, -2].values
    b = df.iloc[1:, -1].values.astype(float)
    return c, A, signs, b

def display_equations(c, A, signs, b):
    print("\n=== Linear Programming Problem ===")
    # Display Objective Function
    print("Maximize: Z = ", end="")
    objective_terms = [f"{c[i]}*x{i+1}" for i in range(len(c))]
    print(" + ".join(objective_terms))
    
    # Display Constraints
    print("\nSubject to:")
    for i in range(len(A)):
        constraint_terms = [f"{A[i][j]}*x{j+1}" for j in range(len(A[i]))]
        print(" + ".join(constraint_terms), f"{signs[i]} {b[i]}")
    print("\n" + "-" * 50)

def print_tableau(tableau, basis_vars, non_basis_vars, variable_names, entering_var=None, leaving_var=None):
    # Prepare the dataframe to display tableau with variable names
    df_tableau = pd.DataFrame(tableau, columns=variable_names + ["RHS"])
    
    # Create a new column for entering and leaving variables
    if entering_var is not None and leaving_var is not None:
        df_tableau.insert(0, "Entering/Leaving", [f"{variable_names[entering_var]} / {variable_names[leaving_var]}" if i == len(tableau) - 1 else "" for i in range(len(tableau))])
    else:
        df_tableau.insert(0, "Entering/Leaving", [""] * len(tableau))
    
    # Replace row indices with variable names
    row_names = [variable_names[basis_vars[i]] if i < len(basis_vars) else f"x{len(variable_names) + 1 + i}" for i in range(len(tableau) - 1)]
    row_names.append('Z')  # For the last row (objective function)
    df_tableau.index = row_names
    
    print("\nCurrent Tableau:")
    print(df_tableau)
    print("Basis Variables:", basis_vars)
    print("Non-Basis Variables:", non_basis_vars)
    print("-" * 50)


def simplex(c, A, b):
    # Convert all inequalities to equations by adding slack variables
    num_constraints, num_variables = A.shape
    tableau = np.hstack([A, np.eye(num_constraints), b.reshape(-1, 1)])
    
    # Initializing the coefficients for the objective function in tableau
    c_row = np.hstack([-c, np.zeros(num_constraints + 1)])
    tableau = np.vstack([tableau, c_row])
    
    # Initial basis (slack variables)
    basis_vars = list(range(num_variables, num_variables + num_constraints))
    non_basis_vars = list(range(num_variables))

    # Variable names for tableau
    variable_names = [f"x{i+1}" for i in range(num_variables)] + [f"s{i+1}" for i in range(num_constraints)]

    print_tableau(tableau, basis_vars, non_basis_vars, variable_names)

    # Iterate through steps of the simplex algorithm
    step = 0
    while True:
        print(f"\n--- Step {step} ---")
        step += 1

        # Check if the solution is optimal
        last_row = tableau[-1, :-1]
        if np.all(last_row >= 0):
            print("Optimal solution found.")
            break

        # Pivot column (entering variable) - the most negative coefficient in the objective row
        pivot_col = np.argmin(last_row)
        print(f"Entering Variable: {variable_names[pivot_col]}")

        # Pivot row (leaving variable)
        if all(tableau[:-1, pivot_col] <= 0):
            print("Unbounded solution.")
            return None

        ratios = tableau[:-1, -1] / tableau[:-1, pivot_col]
        ratios[ratios <= 0] = np.inf
        pivot_row = np.argmin(ratios)
        print(f"Leaving Variable: {variable_names[basis_vars[pivot_row]]}")

        # Pivoting operation
        tableau[pivot_row, :] /= tableau[pivot_row, pivot_col]
        for i in range(len(tableau)):
            if i != pivot_row:
                tableau[i, :] -= tableau[i, pivot_col] * tableau[pivot_row, :]

        # Update basis and non-basis variables
        basis_vars[pivot_row], non_basis_vars[pivot_col] = (
            non_basis_vars[pivot_col],
            basis_vars[pivot_row],
        )

        # Display the tableau after the pivot
        print_tableau(tableau, basis_vars, non_basis_vars, variable_names, entering_var=pivot_col, leaving_var=basis_vars[pivot_row])

    # Extract solution
    solution = np.zeros(num_variables)
    for i in range(num_constraints):
        if basis_vars[i] < num_variables:
            solution[basis_vars[i]] = tableau[i, -1]

    print("\nOptimal Solution:")
    print("Objective Function Value:", tableau[-1, -1])
    print("Variable Values:", solution)
    return solution

if __name__ == "__main__":
    # Use argparse to get the filename from command-line arguments
    parser = argparse.ArgumentParser(description="Simplex Method Solver")
    parser.add_argument("filename", type=str, help="Path to the CSV file containing simplex data")
    args = parser.parse_args()

    # Read data from the CSV file
    c, A, signs, b = read_csv(args.filename)

    # Display the equations
    display_equations(c, A, signs, b)

    print("\n=== Simplex Method ===")
    simplex(c, A, b)
