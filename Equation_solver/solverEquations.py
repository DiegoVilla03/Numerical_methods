import time
import numpy as np
import matplotlib as plt
import sympy as sp
from sympy import lambdify

x = sp.Symbol('x')


# * Definition of the methods ude to solve ecuations in the form f(x)=0
def newton_raphson(f, Df, D2f, x0, epsilon, max_iter):
    xn = x0
    for n in range(0, max_iter):
        fxn = f(xn)
        if abs(fxn) < epsilon:
            print('Found solution after', n, 'iterations.')
            return xn
        Dfxn = Df(xn)
        D2fxn = D2f(xn)
        den = Dfxn ** 2 - fxn * D2fxn
        if den == 0:
            print("Zero denominator, no solution found")
            return None
        if Dfxn == 0:
            print('Zero derivative. No solution found.')
            return None
        xn = xn - fxn * Dfxn / den
    print('Exceeded maximum iterations. No solution found.')
    return None


def my_bisection(f, a, b, tol):
    if np.sign(f(a)) == np.sin(f(b)):
        print("The scalars a and b do not bound a root")
        return None

    else:
        m = (a + b) / 2

        if np.abs(f(m)) < tol:
            return m
        elif np.sign(f(a)) == np.sign(f(m)):
            return my_bisection(f, m, b, tol)
        elif np.sign(f(b)) == np.sign(f(m)):
            return my_bisection(f, a, m, tol)


def newton(f, f1, x0, epsilon, max_iter):
    xn = x0
    for n in range(0, max_iter):
        fxn = f(xn)
        if abs(fxn) < epsilon:
            print('Found solution after', n, 'iterations.')
            return xn
        f1xn = f1(xn)
        if f1xn == 0:
            print('Zero derivative. No solution found.')
            return None

        xn = xn - fxn / f1xn
    print('Exceeded maximum iterations. No solution found.')
    return None


def secante(f, x0, x1, tol, max_iter):
    for n in range(0, max_iter):
        den = (f(x1) - f(x0))
        if den == 0:
            print("Zero Denominator no solution found")
            return None

        x2 = x1 - (f(x1) * (x1 - x0)) / den

        if abs(x2 - x1) < tol:
            print("Found Solution after", n, "iterations")
            return x2
        x0 = x1
        x1 = x2


def ToSolve():
    confirm = True
    print("\033[0;31m" + "\033[1m")
    print("rules to imput the function")
    print("polynomial functions like x^n is equally to x**n")
    print("trigonometric functions are as follow sin(x), cos(x), tan(x)")
    print("logarithmic functions like ln(x) is equally to use log(x)")
    print("the use of parenthesis is highly recommended ")
    print("the use of any other special functions than the specified has no support "
          "and subsequent errors are not dev team responsibility")
    print("\033[0m")
    while confirm == True:
        f = input("Input the function to solve in the specified format = ")
        print("are you sure to input this equation  = ", f)
        conf = int(input("confirm 1, cancel 0 "))
        if conf == 1:
            confirm = False
        else:
            confirm = True

    f_prime = sp.diff(f, x)
    f_2prime = sp.diff(f_prime, x)

    F = lambdify(x, f)
    F_prime = lambdify(x, f_prime, modules='numpy')
    F_2prime = lambdify(x, f_2prime, modules='numpy')
    return F, F_prime, F_2prime


def interval():
    a = b = 0
    print("give an interval to evaluate the function")
    print("note that to imput real numbers as interval please use . as decimal separator")
    print("please keep in mind that a must be greater than b")
    while a >= b:
        a = float(input("beginning of the interval = "))
        b = float(input("End of the interval = "))
        if a >= b:
            print("interval not valid please try again")
    return a, b


def tolerance():
    tol = 0
    while tol <= 0:
        tol = int(input("give the number of decimals to aproximate"))

        if tol <= 0:
            print("tolerance no valid please try again")

    return tol


def conditions():
    print("please give a initial guess to the solution of the equation")
    print("keep in mind that the better the aproximation is the more probable the method will give an answer \n")

    guess = float(input("give the initial value  "))

    print("please enter the maximum number of iteration the method will do")
    print("also consider not to exceed more than 20 iterations otherwise is no considered safe")

    limit = int(input("maximum iteration = "))

    return guess, limit


def menu():
    ans = True
    while ans:
        print("\033[0;32m" + "\033[1m")
        print("Options:")
        print("1 Bisection")
        print("2 Newton")
        print("3 Newton-Raphson")
        print("4 Secant Method")
        print("5 End Session")
        print("\033[0m")
        ans = input("What Method would you like to use")
        if ans == "1":
            print("\nbisection")
            f, _, _ = ToSolve()
            a, b = interval()
            tol = float("1e-" + str(tolerance()))
            result = my_bisection(f, a, b, tol)
            print("Solution calculated:", result, "\n")
        elif ans == "2":
            print("\nNewton")
            f, f1, _ = ToSolve()
            tol = float("1e-" + str(tolerance()))
            guess, limit = conditions()
            result = newton(f, f1, guess, tol, limit)
            print("solution calculated:", result, "\n")
        elif ans == "3":
            print("\nNewton Raphson")
            f, f1, f2 = ToSolve()
            tol = float("1e-" + str(tolerance()))
            guess, limit = conditions()
            result = newton_raphson(f, f1, f2, guess, tol, limit)
            print("Solution calculated:", result, "\n")
        elif ans == "4":
            print("\n Secant Method")
            f, _, _ = ToSolve()
            tol = float("1e-" + str(tolerance()))
            guess, limit = conditions()
            result = secante(f, guess, guess + 0.1, tol, limit)
            print("Solution Calculated: ", result, "\n")
        elif ans == "5":
            ans = None
        else:
            print("\n Not Valid Choice Try again")


print("\033[1;36m" + "\033[1m")
print("                                            _______________________")
print("  _______________________-------------------                       `|")
print('/:--__                                                              |')
print('||< > |                                   ___________________________/')
print("| \__/_________________-------------------                         |")
print("|                                                                  |")
print("|                       EQUATION SOLVER                             |")
print(" |                                                                  |")
print(" |      this application use different methods to solve              |")
print(" |      equations in the form of f(x) = 0                           |")
print(" |                                                                  |")
print(" |      WARNING: The app requires some well know mandatory          |")
print(" |      arguments to work correctly please ensure that your        |")
print(" |      inputs are correct otherwise the app will let you know      |")
print("|                                              ____________________|_")
print("|  ___________________-------------------------                      `|")
print("|/`--_                                                                |")
print("||[ ]||                                            ___________________/")
print(" \===/___________________--------------------------")
print("\033[0m")

time.sleep(1)

menu()

print("\033[1;36m" + "\033[1m")
print("                       _ _                ")
print("                      | | |               ")
print("  __ _  ___   ___   __| | |__  _   _  ___ ")
print(" / _` |/ _ \ / _ \ / _` | '_ \| | | |/ _ |")
print("| (_| | (_) | (_) | (_| | |_) | |_| |  __/")
print(" \__, |\___/ \___/ \__,_|_.__/ \__, |\___|")
print("  __/ |                         __/ |    ")
print(" |___/                         |___/      ")
print("\033[0m")
