import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse


def read_csv(file_path):
    """
    Reads from a csv file and stores the values as arrays
    """
    df = pd.read_csv(file_path, header=None)
    # Objective Function
    c = df.iloc[0, :-1].dropna().values.astype(float)
    # Constraints
    A = df.iloc[1:, :-2].dropna(axis=1).values.astype(float)
    signs = df.iloc[1:, -2].values
    b = df.iloc[1:, -1].values.astype(float)
    return c, A, signs, b



def read_excel(file_path):
    """
    Reads from a excel (xlsx) file and stores the values as arrays
    """
    df = pd.read_excel(file_path, header=None)
    # Objective Function
    c = df.iloc[0, :-1].dropna().values.astype(float)
    # Constraints
    A = df.iloc[1:, :-2].dropna(axis=1).values.astype(float)
    signs = df.iloc[1:, -2].values
    b = df.iloc[1:, -1].values.astype(float)
    return c, A, signs, b





def display_equations(c, A, signs, b):
    """
    Prints in the terminal the equations passed as arguments
    """
    print("\n=== Linear Programming Problem ===")
    # Display Objective Function
    print("Maximize: Z = ", end="")
    objective_terms = [f"{c[i]}x{i+1}" for i in range(len(c))]
    print(" + ".join(objective_terms))
    
    # Display Constraints
    print("\nSubject to:")
    for i in range(len(A)):
        constraint_terms = [f"{A[i][j]}x{j+1}" for j in range(len(A[i]))]
        print(" + ".join(constraint_terms), f"{signs[i]} {b[i]}")
    print("\n" + "-" * 50)

def print_tableau(tableau, header):
    """
    Prints the current simplex tableau in a readable format with a fixed header.
    """
    rows, cols = tableau.shape

    # Print the header
    print(f"{'':<10}", end="")
    for col in header:
        print(f"{col:<10}", end="")
    print("\n" + "-" * (10 + 10 * cols))

    # Print each row of the tableau
    for i, row in enumerate(tableau[:-1]):
        row_label = header[i]  # Use the fixed header for row labels
        print(f"{row_label:<10}", end="")
        for val in row:
            print(f"{val:<10.2f}", end="")
        print()
    
    # Print the objective function (Z-row)
    print(f"{'Z':<10}", end="")
    for val in tableau[-1]:
        print(f"{val:<10.2f}", end="")
    print("\n")


def plot_feasible_region(c, A, b, solution):
    # Set a finer scale for a more zoomed-in view
    x_values = np.linspace(-2, 12, 500)  # Extended range with more precision
    
    plt.figure(figsize=(10, 8))
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Feasible Region, Constraints, and Optimal Solution')
    
    # Plot each constraint line and the intersection points with axes
    for i in range(len(A)):
        if A[i][1] != 0:  # Regular line (not vertical)
            y_values = (b[i] - A[i][0] * x_values) / A[i][1]
            plt.plot(x_values, y_values, label=f"Constraint {i+1}")
            
            # Calculate intersection with x1 and x2 axes
            x_intercept = b[i] / A[i][0] if A[i][0] != 0 else None
            y_intercept = b[i] / A[i][1] if A[i][1] != 0 else None
            
            # Mark intersections on the plot
            if x_intercept is not None and x_intercept >= 0:
                plt.plot(x_intercept, 0, 'ro')
                plt.text(x_intercept, 0, f'({x_intercept:.2f}, 0)', fontsize=10, color='red')
            if y_intercept is not None and y_intercept >= 0:
                plt.plot(0, y_intercept, 'bo')
                plt.text(0, y_intercept, f'(0, {y_intercept:.2f})', fontsize=10, color='blue')
        
        else:  # Vertical line (x2 coefficient is zero)
            x_value = b[i] / A[i][0]
            plt.axvline(x=x_value, label=f"Constraint {i+1}")
            plt.plot(x_value, 0, 'ro')
            plt.text(x_value, 0, f'({x_value:.2f}, 0)', fontsize=10, color='red')
    
    # Shade the feasible region
    feasible_region = np.minimum.reduce(
        [(b[i] - A[i][0] * x_values) / A[i][1] if A[i][1] != 0 else 12 for i in range(len(A))]
    )
    plt.fill_between(x_values, 0, feasible_region, where=feasible_region >= 0, color='gray', alpha=0.3)

    # Plot the objective function (assuming we want to maximize Z)
    if c[1] != 0:
        z_value = c @ solution  # Compute the value of the objective function at the solution
        y_obj = (z_value - c[0] * x_values) / c[1]
        plt.plot(x_values, y_obj, '--', label="Objective Function")

    # Plot the optimal solution point
    plt.plot(solution[0], solution[1], 'go', markersize=10, label="Optimal Solution")
    plt.text(solution[0], solution[1], f'Optimum ({solution[0]:.2f}, {solution[1]:.2f})', fontsize=12, color='green')

    # Draw X-axis and Y-axis with a smaller scale
    plt.axhline(0, color='black', linewidth=1.5)  # X-axis
    plt.axvline(0, color='black', linewidth=1.5)  # Y-axis

    # Adjust plot limits for a finer view
    plt.xlim(-2, 12)
    plt.ylim(-2, 12)
    plt.xticks(np.arange(-2, 13, 1))
    plt.yticks(np.arange(-2, 13, 1))
    
    plt.legend()
    plt.grid(True)

    # Show the plot
    print("Displaying plot... Close the window to continue.")
    plt.show()  # Use blocking mode to display the window

def simplex(c, A, b):
    num_constraints, num_variables = A.shape
    tableau = np.hstack([A, np.eye(num_constraints), b.reshape(-1, 1)])

    # Add objective function row
    c_row = np.hstack([-c, np.zeros(num_constraints + 1)])
    tableau = np.vstack([tableau, c_row])

    # Track basis and non-basis variables
    basis_vars = list(range(num_variables, num_variables + num_constraints))
    non_basis_vars = list(range(num_variables))

    # Create fixed header
    header = [f"x{i + 1}" for i in range(num_variables)] + \
             [f"s{i + 1}" for i in range(num_constraints)] + ["P0"]

    step = 0
    while True:
        print(f"\n--- Step {step} ---")
        step += 1

        # Print tableau with fixed header
        print_tableau(tableau, header)

        # Check if the solution is optimal
        last_row = tableau[-1, :-1]
        if np.all(last_row >= 0):
            print("Optimal solution found.")
            break

        # Check if the solution is infinite
        if np.any(np.isclose(last_row, 0)):  # Objective row contains a zero coefficient
            # Find constraints parallel to the objective function
            for i, row in enumerate(A):
                if np.isclose(row @ c, c @ c):  # Check for parallelism
                    print(f"Infinite solutions detected due to constraint {i + 1} being parallel to the objective function.")
                    print(f"The feasible region extends infinitely along the constraint starting at point: {b[i]}.")
                    break

        # Choose pivot column (entering variable)
        pivot_col = np.argmin(last_row)
        print(f"Entering Variable: x{pivot_col+1}")

        # Check for unbounded solution
        valid_rows = tableau[:-1, pivot_col] > 0
        if not np.any(valid_rows):
            print("Unbounded solution detected.")
            return None

        # Choose pivot row (leaving variable)
        ratios = tableau[:-1, -1][valid_rows] / tableau[:-1, pivot_col][valid_rows]
        pivot_row = np.where(valid_rows)[0][np.argmin(ratios)]
        print(f"Leaving Variable: {header[pivot_row]}")

        # Perform pivoting
        tableau[pivot_row, :] /= tableau[pivot_row, pivot_col]
        for i in range(len(tableau)):
            if i != pivot_row:
                tableau[i, :] -= tableau[i, pivot_col] * tableau[pivot_row, :]

        # Update basis
        basis_vars[pivot_row], non_basis_vars[pivot_col] = non_basis_vars[pivot_col], basis_vars[pivot_row]

    # Extract solution
    solution = np.zeros(num_variables)
    for i, var in enumerate(basis_vars):
        if var < num_variables:
            solution[var] = tableau[i, -1]

    print("\nOptimal Solution:")
    print("Objective Function Value:", tableau[-1, -1])
    print("Variable Values:", solution)
    return solution


if __name__ == "__main__":
    # Use argparse to get the filename from command-line arguments
    parser = argparse.ArgumentParser(description="Simplex Method Solver")
    parser.add_argument(
        "-csv", action="store_true", help="Flag to read data from a CSV file"
    )
    parser.add_argument("filename", type=str, help="Path to the CSV file containing simplex data")
    args = parser.parse_args()


    # Flags for selecting file type
    if args.csv:
        c, A, signs, b = read_csv(args.filename)
    else:
        c, A, signs, b = read_excel(args.filename)





    # Display the equations
    display_equations(c, A, signs, b)

    print("\n=== Simplex Method ===")
    solution = simplex(c, A, b)


    if solution  is not None and len(c) == 2:
        plot_feasible_region(c, A, b, solution)
