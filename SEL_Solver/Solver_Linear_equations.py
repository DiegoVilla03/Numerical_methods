# ************************************************************
# Author: Diego Villarreal De La Cerda
# Date: 16/10/2023
# Description: System of Linear equations Solver
# ************************************************************

from pprint import pprint
import sympy as sp
import numpy as np

def SL_solve(A, B, C):
    detA = np.linalg.det(A)
    print("the determinant of the system is")
    print(detA)
    if np.abs(detA) >= 1e-5:
        print("the system has a unique solution")
        x = np.linalg.solve(A, B)
        print("the result is: ")
        pprint(x)
    else:
        while np.any((0 < np.abs(C)) & (np.abs(C) < 1)):
            C = 10 * C

        x = C.rref()[0]
        NS = NoSolution(x)

        if NS == True:
            print("the system has no solution")
            pprint(x)
        else:
            print("the system has infinite solutions")
            print("the result is: ")
            pprint(x)
            print("\n")


def InValue():
    print("how many equations and variables does the system have?")
    n = int(input())
    A = np.zeros((n, n))
    B = np.zeros(n)

    for i in range(0, n):
        row = np.array([float(x) for x in input(f"Enter values for row {i + 1} separated by spaces: ").split()])
        A[i, :] = row

    col = np.array([float(x) for x in input(f"input the solution vector separated by spaces: ").split()])
    B[:] = col

    sympy_matrix = sp.Matrix(A)
    sympy_vector = sp.Matrix(B)
    C = sp.Matrix.hstack(sympy_matrix, sympy_vector)
    pprint(C)
    print("\n")

    return A, B, C


def NoSolution(C):
    last = C.row(-1)
    allzero = all(elemento == 0 for elemento in last)
    if allzero:
        return False
    else:
        return True


def menu():
    print("System of linear equation Solver")
    ans = True
    while ans:
        print("1 solve system")
        print("2 Exit")
        ans = int(input("please select what would you like to do"))
        if ans == 1:
            A, B, C = InValue()
            SL_solve(A, B, C)
        elif ans == 2:
            ans = None
        else:
            print("\n Not Valid Choose Try Again")


menu()
