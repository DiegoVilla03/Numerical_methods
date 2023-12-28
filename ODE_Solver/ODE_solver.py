# ************************************************************
# Author: Diego Villarreal De La Cerda
# Date: 
# Description: diferential equations Solver
# ************************************************************

import numpy as np
import matplotlib.pyplot as plt
import functions as F
from rich.table import Table
from rich.console import Console


def euler(f, t0, tn, n, y0):
    """
    Approximate the solution of a first-order ordinary differential equation using the Euler method.

    Parameters:
    - f: Function representing the differential equation (dy/dt = f(t, y)).
    - t0: Initial time.
    - tn: Final time.
    - n: Number of intervals.
    - y0: Initial value of y at t0.

    Returns:
    - y: Array containing the approximate solution at each time step.
    """
    h = abs(tn - t0) / n
    t = np.linspace(t0, tn, n + 1)  # Linespace will help us to make tSol [the array with all the intervals]
    y = np.zeros(n + 1)  # remember that tSol and ySol need to have the same size
    y[0] = y0
    for k in range(0, n):
        y[k + 1] = y[k] + h * f(t[k], y[k])
    return y


def RK4 (f, t0, tn, n, y0):
    """
    Approximate the solution of a first-order ordinary differential equation using the Runge-Kutta 4th Order method.

    Parameters:
    - f: Function representing the differential equation (dy/dt = f(t, y)).
    - t0: Initial time.
    - tn: Final time.
    - n: Number of intervals.
    - y0: Initial value of y at t0.

    Returns:
    - y: Array containing the approximate solution at each time step.
    """
    h = abs(tn - t0) / n
    t = np.linspace(t0, tn, n + 1)  #* Linespace will help us to make tSol [the array with all the intervals]
    y = np.zeros(n + 1)  #* remember that tSol and ySol need to have the same size
    y[0] = y0
    
    for k in range(0, n):
        k1 = f(t[k], y[k])
        k2 = f(t[k] + (h / 2), y[k] + k1 * (h / 2) )
        k3 = f(t[k] + (h / 2), y[k] + k2 * (h / 2) )
        k4 = f(t[k] + h, y[k] + k3*h)
        y[k+1] = y[k] + (h / 6) * (k1 +( 2 * k2) +( 2 * k3) + k4)
    return y


def euler_r(f, t0, tn, n, y0, r):
    """
    Approximate the solution of a first-order ordinary differential equation with a parameter using the Euler method.

    Parameters:
    - f: Function representing the differential equation (dy/dt = f(t, y, r)).
    - t0: Initial time.
    - tn: Final time.
    - n: Number of intervals.
    - y0: Initial value of y at t0.
    - r: Additional parameter in the differential equation with variable r parameter.

    Returns:
    - y: Array containing the approximate solution at each time step.
    """
    h = abs(tn - t0) / n
    t = np.linspace(t0, tn, n + 1)  #* Linespace will help us to make tSol [the array with all the intervals]
    y = np.zeros(n + 1)  #* remember that tSol and ySol need to have the same size
    y[0] = y0
    for k in range(0, n):
        y[k + 1] = y[k] + h * f(t[k], y[k], r)
    return y


def RK4_r (f, t0, tn, n, y0, r):
    """
    Approximate the solution of a first-order ordinary differential equation with a parameter using the Runge-Kutta 4th Order method.

    Parameters:
    - f: Function representing the differential equation (dy/dt = f(t, y, r)).
    - t0: Initial time.
    - tn: Final time.
    - n: Number of intervals.
    - y0: Initial value of y at t0.
    - r: Additional parameter in the differential equation with variable r .

    Returns:
    - y: Array containing the approximate solution at each time step.
    """
    h = abs(tn - t0) / n
    t = np.linspace(t0, tn, n + 1)  #* Linespace will help us to make tSol [the array with all the intervals]
    y = np.zeros(n + 1)  #* remember that tSol and ySol need to have the same size
    y[0] = y0
    
    for k in range(0, n):
        k1 = f(t[k], y[k], r)
        k2 = f(t[k] + (h / 2), y[k] + k1 * (h / 2) , r)
        k3 = f(t[k] + (h / 2), y[k] + k2 * (h / 2) , r)
        k4 = f(t[k] + h, y[k] + k3*h, r)
        y[k+1] = y[k] + (h / 6) * (k1 +( 2 * k2) +( 2 * k3) + k4)
    return y


def plot_results(x, y_exact, ye, yrk4, fig_title):
    """
    Plot the results of the differential equation solvers.

    Parameters:
    - x: Array of x values.
    - y_exact: Exact solution array.
    - ye: Euler method solution array.
    - yrk4: Runge-Kutta 4th Order method solution array.
    - fig_title: Title for the plot.
    """
    fig = plt.figure()
    fig.suptitle(fig_title, fontsize=12)
    plt.plot(x, ye, 'bo', label = 'Euler')
    plt.plot(x, yrk4, 'g-o', label  = 'Runge Kutta 4')
    plt.plot(x, y_exact, 'r--', label = 'Exact solution')
    plt.grid()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()



def plot_results_non(x, ye, yrk4, fig_title):
    """
    Plot the results of the differential equation solvers without the exact solution.

    Parameters:
    - x: Array of x values.
    - ye: Euler method solution array.
    - yrk4: Runge-Kutta 4th Order method solution array.
    - fig_title: Title for the plot.
    """
    fig = plt.figure()
    fig.suptitle(fig_title, fontsize=12)
    plt.plot(x, ye, 'bo', label = 'Euler')
    plt.plot(x, yrk4, 'g-o', label  = 'Runge Kutta 4')
    plt.grid()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()


def P1():
    """
    Function to handle Practice 1 exercises.

    Prints a menu for six exercises done in class and allows the user to choose which exercise to display.
    For each exercise, it sets initial conditions, solves the differential equation using Euler and Runge-Kutta methods,
    calculates the exact solution, and plots the results.

    Exercises:
    1. Exercise 1.1
    2. Exercise 1.2
    3. Exercise 1.3
    4. Exercise 1.4
    5. Exercise 1.5
    6. Exercise 1.6

    Returns:
    - None
    """
    print('This section belongs to Practice 1')
    print('the 6 excercises done in class')

    ans1 = True
    while ans1:
        print('1. Solve Excercise 1')
        print('2. Solve Excercise 2')
        print('3. Solve Excercise 3')
        print('4. Solve Excercise 4')
        print('5. Solve Excercise 5')
        print('6. Solve Excercise 6')
        print('7. Return to main Menu')
        
        ans1 = int(input('please select what excercise would you like to see'))
        if ans1 == 1:
            print('Excercise 1.1')
            t0 = 0
            tend = 60
            y0 = 4
            n = 10 # int, numero de subintervalos

            x = np.linspace(t0,tend,n+1) #en aplicado será tiempo
            y_exact = F.E1_1(x)
            ye = euler(F.f1_1, t0, tend, n, y0)
            yrk = RK4(F.f1_1, t0, tend, n, y0)

            plot_results(x, y_exact, ye, yrk, "Exercise 1.1")
        elif ans1 == 2:
            print('Excercise 1.2')
            t0 = 1
            tend = 600
            y0 = 0
            n = 10 #### int, numero de subintervalos

            x = np.linspace(t0,tend,n+1) #en aplicado será tiempo
            y_exact = F.E1_2(x)
            ye = euler(F.f1_2, t0, tend, n, y0)
            yrk = RK4(F.f1_2, t0, tend, n, y0)

            plot_results(x, y_exact, ye, yrk, "Exercise 1.2")
        elif ans1 == 3:
            print('Excercise 1.3')
            t0 = 1
            tend = 10
            y0 = 0
            n = 10 #### int, numero de subintervalos

            x = np.linspace(t0,tend,n+1) #en aplicado será tiempo
            y_exact = F.E1_3(x)
            ye = euler(F.f1_3, t0, tend, n, y0)
            yrk = RK4(F.f1_3, t0, tend, n, y0)

            plot_results(x, y_exact, ye, yrk, "Exercise 1.3")
        elif ans1 == 4:
            print('Excercise 1.4')
            t0 = 0
            tend = 55800
            y0 = 100
            n = 10 #### int, numero de subintervalos

            x = np.linspace(t0,tend,n+1) #en aplicado será tiempo
            y_exact = F.E1_4(x)
            ye = euler(F.f1_4, t0, tend, n, y0)
            yrk = RK4(F.f1_4, t0, tend, n, y0)

            plot_results(x, y_exact, ye, yrk, "Exercise 1.4")
        elif ans1 == 5:
            print('Excercise 1.5')
            t0 = 0
            tend = 60
            y0 = 300
            n = 10 #### int, numero de subintervalos

            x = np.linspace(t0,tend,n+1) #en aplicado será tiempo
            y_exact = F.E1_5(x)
            ye = euler(F.f1_5, t0, tend, n, y0)
            yrk = RK4(F.f1_5, t0, tend, n, y0)

            plot_results(x, y_exact, ye, yrk, "Exercise 1.5")
        elif ans1 == 6:
            print('Excercise 1.6')
            t0 = 0
            tend = 600
            y0 = 50
            n = 10 #### int, numero de subintervalos

            x = np.linspace(t0,tend,n+1) #en aplicado será tiempo
            y_exact = F.E1_6(x)
            ye = euler(F.f1_6, t0, tend, n, y0)
            yrk = RK4(F.f1_6, t0, tend, n, y0)

            plot_results(x, y_exact, ye, yrk, "Exercise 1.6")
        elif ans1 == 7:
            ans1= None
        else:
            print("\n Not Valid Choose Try Again")    
        

def P2():
    """
    Function to handle Practice 2 exercises.

    Prints a menu for three exercises and allows the user to choose which exercise to display.
    For each exercise, it sets initial conditions, solves the differential equation using Euler and Runge-Kutta methods,
    calculates the exact solution, plots the results, and displays a table summarizing the solutions.

    Exercises:
    1. Exercise 2.1
    2. Exercise 2.2
    3. Exercise 2.3

    Returns:
    - None
    """
    print('This section belongs to Practice 2')

    ans2 = True
    while ans2:
        print('1. Solve Excercise 2.1')
        print('2. Solve Excercise 2.2')
        print('3. Solve Excercise 2.3')
        print('4. Return to main menu')

        ans2 = int(input('What would you like to do '))
        if ans2 == 1:
            print('Excercise 2.1')
            t0 = 0
            tend = [1,2,3,4,5,6]
            y0 = 5000
            
            table1 = Table()
            table1.add_column('años')
            table1.add_column('meses')
            table1.add_column('Euler')
            table1.add_column('RK4')
            table1.add_column('Real Soltion')
            
            for i in range(0,6):
                n=(tend[i]*12)
                x = np.linspace(t0,tend[i],n+1)
                y_exact = F.E2_1(x)
                ye = euler(F.f2_1, t0, tend[i], n, y0)
                yrk = RK4(F.f2_1, t0, tend[i], n, y0)
                
                plot_results(x, y_exact, ye, yrk, 'Solution claculated')
                table1.add_row(str(i+1), str(n), str(ye[-1]), str(yrk[-1]), str(y_exact[-1]))
            
            console = Console()  
            console.print(table1)
        
        elif ans2 == 2:
            print('Excercise 2.2')
            t0 = 0
            tend = 5
            y0 = 5000
            n = 60
            
            table2 = Table()
            table2.add_column('interest %')
            table2.add_column('Euler')
            table2.add_column('RK4')
            table2.add_column('Real Soltion')
            
            for r in np.arange(0.01,0.07,0.01):
                x = np.linspace(t0, tend, n+1)
                y_exact = F.E2_2(x,r)
                ye = euler_r(F.f2_2, t0, tend, n,y0, r)
                yrk = RK4_r(F.f2_2,t0,tend,n,y0,r)
                
                plot_results(x, y_exact, ye, yrk, 'Solution claculated')
                table2.add_row(str(round(r,2)), str(ye[-1]), str(yrk[-1]), str(y_exact[-1]))
            
            console = Console()
            console.print(table2)    
        
        elif ans2 == 3:
            print('Excercise 2.3')
            t0 = 0
            tend = [5,10,15]
            y0 = 10000
            
            table3 = Table()
            table3.add_column('años')
            table3.add_column('Semestres')
            table3.add_column('Euler')
            table3.add_column('RK4')
            table3.add_column('Real Soltion')
            
            for i in range(0,3):
                n=(tend[i]*2)
                x = np.linspace(t0,tend[i],n+1)
                y_exact = F.E2_3(x)
                ye = euler(F.f2_3, t0, tend[i], n, y0)
                yrk = RK4(F.f2_3, t0, tend[i], n, y0)
                
                
                
                plot_results(x, y_exact, ye, yrk, 'Solution claculated')
                table3.add_row(str(tend[i]), str(n), str(ye[-1]), str(yrk[-1]), str(y_exact[-1]))
            
            console = Console()  
            console.print(table3)
            
        elif ans2 == 4:
            ans2 = None
        else:
            print('Not Valid Option Try again')
        
        
def P3():
    """
    Function to handle Exercise 3.

    Prints information about Exercise 3, sets initial conditions, solves the differential equation using Euler
    and Runge-Kutta methods, and plots the results without displaying the exact solution.

    Returns:
    - None
    """
    print('Excercise 3')
    t0 = 0
    y0 = 0.01
    tend = 2/(y0)
    h = tend/500
    n = int((tend-t0)/h)
    
    x = np.linspace(t0,tend,n+1)
    ye = euler(F.f3_1, t0, tend, n, y0)
    yrk = RK4(F.f3_1, t0, tend, n, y0)
    plot_results_non(x,ye,yrk, 'Excercise 3')
    
    
def P4():
    """
    Function to handle Practice 4 exercises.

    Prints a menu for two exercises and allows the user to choose which exercise to display.
    For each exercise, it sets initial conditions, solves the differential equation using Euler and Runge-Kutta methods,
    calculates the exact solution, plots the results, and displays tables summarizing errors for different step sizes.

    Exercises:
    1. Exercise 4.1
    2. Exercise 4.2

    Returns:
    - None
    """
    print('this section belongs to practice 4')
    
    ans3 = True
    while ans3:
        print('1. Excercise 4.1')
        print('2. Excercise 4.2')
        print('3. Return yo main menu')
        
        ans3 = int(input('please select what would you like to do'))
        
        if ans3 == 1:
            print('Excercise 4.1')
            t0 = 0
            y0 = 1
            tend = 1
            h = [0.1,0.05]
            n1 = int((tend-t0)/h[0])
            n2 = int((tend-t0)/h[1])
            
            x = np.linspace(t0,tend,n1+1)
            y_exact = F.E4_1(x)
            ye = euler(F.f4_1, t0, tend, n1, y0)
            yrk = RK4(F.f4_1, t0, tend, n1, y0)
            
            plot_results(x, y_exact, ye, yrk, 'Excercise 4.1 h = 0.1')
            
            abs_error_euler = np.abs(y_exact-ye)
            rel_error_euler = abs_error_euler/y_exact
            
            abs_error_rk = np.abs(y_exact - yrk)
            rel_error_rk = abs_error_rk/y_exact
            
            Table4_1_e1 = Table()
            Table4_1_e1.add_column('xn')
            Table4_1_e1.add_column('euler solution')
            Table4_1_e1.add_column('Real Solution')
            Table4_1_e1.add_column('Absolute Error')
            Table4_1_e1.add_column('Relative Error')
            
            Table4_1_R1 = Table()
            Table4_1_R1.add_column('xn')
            Table4_1_R1.add_column('RK4 solution')
            Table4_1_R1.add_column('Real Solution')
            Table4_1_R1.add_column('Absolute Error')
            Table4_1_R1.add_column('Relative Error')
            
            for i in range(len(x)):
                Table4_1_e1.add_row(str(round(x[i],2)), str(ye[i]), str(y_exact[i]), str(round(abs_error_euler[i],16)), str(round(rel_error_euler[i],16)))
                Table4_1_R1.add_row(str(round(x[i],2)), str(yrk[i]), str(y_exact[i]), str(round(abs_error_rk[i],16)), str(round(rel_error_rk[i], 16)))
            
            console = Console()
            console.print(Table4_1_e1)
            console.print(Table4_1_R1)
            
            x = np.linspace(t0,tend,n2+1)
            y_exact = F.E4_1(x)
            ye = euler(F.f4_1, t0, tend, n2, y0)
            yrk = RK4(F.f4_1, t0, tend, n2, y0)
            
            
            plot_results(x, y_exact, ye, yrk, 'Excercise 4.1 h = 0.05')
            abs_error_euler = np.abs(y_exact-ye)
            rel_error_euler = abs_error_euler/y_exact
            
            abs_error_rk = np.abs(y_exact - yrk)
            rel_error_rk = abs_error_rk/y_exact
            
            Table4_1_e2 = Table()
            Table4_1_e2.add_column('xn')
            Table4_1_e2.add_column('euler solution')
            Table4_1_e2.add_column('Real Solution')
            Table4_1_e2.add_column('Absolute Error')
            Table4_1_e2.add_column('Relative Error')
            
            Table4_1_R2 = Table()
            Table4_1_R2.add_column('xn')
            Table4_1_R2.add_column('RK4 solution')
            Table4_1_R2.add_column('Real Solution')
            Table4_1_R2.add_column('Absolute Error')
            Table4_1_R2.add_column('Relative Error')
            
            for i in range(len(x)):
                Table4_1_e2.add_row(str(round(x[i],2)), str(ye[i]), str(y_exact[i]), str(round(abs_error_euler[i],16)), str(round(rel_error_euler[i],16)))
                Table4_1_R2.add_row(str(round(x[i],2)), str(yrk[i]), str(y_exact[i]), str(round(abs_error_rk[i],16)), str(round(rel_error_rk[i], 16)))
            
            console = Console()
            console.print(Table4_1_e2)
            console.print(Table4_1_R2)
            
        elif ans3 == 2:
            print('Excercise 4.2')
            t0 = 0
            y0 = 1
            tend = 10
            h = [0.25,0.1,0.05]
            n1 = int((tend-t0)/h[0])
            n2 = int((tend-t0)/h[1])
            n3 = int((tend-t0)/h[2])
            
            x = np.linspace(t0,tend,n1+1)
            y_exact = F.E4_2(x)
            ye = euler(F.f4_2, t0, tend, n1, y0)
            yrk = RK4(F.f4_2, t0, tend, n1, y0)
            
            plot_results(x, y_exact, ye, yrk, 'Excercise 4.2, h = 0.25')
            
            x = np.linspace(t0,tend,n2+1)
            y_exact = F.E4_2(x)
            ye = euler(F.f4_2, t0, tend, n2, y0)
            yrk = RK4(F.f4_2, t0, tend, n2, y0)
            
            plot_results(x, y_exact, ye, yrk, 'Excercise 4.2, h = 0.1')
            
            x = np.linspace(t0,tend,n3+1)
            y_exact = F.E4_2(x)
            ye = euler(F.f4_2, t0, tend, n3, y0)
            yrk = RK4(F.f4_2, t0, tend, n3, y0)
            
            plot_results(x, y_exact, ye, yrk, 'Excercise 4.2, h = 0.05')
        
        elif ans3 == 3:
            ans3 = None
        
        else:
            print('Not Valid option, Try again')


def P5():
    """
    Function to handle Practice 5 exercises.

    Prints a menu for five exercises and allows the user to choose which exercise to display.
    For each exercise, it sets initial conditions, solves the differential equation using Euler and Runge-Kutta methods,
    calculates the exact solution, plots the results, and displays tables summarizing errors for different step sizes.

    Exercises:
    1. Exercise 5.1 (a-d)
    2. Exercise 5.2
    3. Exercise 5.3
    4. Exercise 5.4
    5. Exercise 5.5
    6. Return to Main Menu

    Returns:
    - None
    """
    
    ans5 = True
    while ans5:
        print('1. Excercise 5.1')
        print('2. Excercise 5.2')
        print('3. Excercise 5.3')
        print('4. Excercise 5.4')
        print('5. Excercise 5.5')
        print('6 Return to Main Menu')
        
        ans5 = int(input('What Would you like to do'))
        if ans5 == 1:
            print('a)')
            t0 = 0
            y0 = 0
            tend = 1
            n = 2
            
            x = np.linspace(t0,tend,n+1)
            y_exact = F.E5_1_a(x)
            ye = euler(F.f5_1_a, t0, tend, n, y0)
            yrk = RK4(F.f5_1_a, t0, tend, n, y0)
            
            plot_results(x, y_exact, ye, yrk, 'Excercise 5.1 a)')
            
            print('Euler :', ye[-1], 'RK4 :', yrk[-1], 'Real :', y_exact[-1])
            
            print('b)')
            t0 = 2
            y0 = 1
            tend = 3
            n = 2
            
            x = np.linspace(t0,tend,n+1)
            y_exact = F.E5_1_b(x)
            ye = euler(F.f5_1_b, t0, tend, n, y0)
            yrk = RK4(F.f5_1_b, t0, tend, n, y0)
            
            plot_results(x, y_exact, ye, yrk, 'Excercise 5.1 b)')
            
            print('Euler :', ye[-1], 'RK4 :', yrk[-1], 'Real :', y_exact[-1])
            
            print('c)')
            t0 = 1
            y0 = 2
            tend = 2
            n = 4
            
            x = np.linspace(t0,tend,n+1)
            y_exact = F.E5_1_c(x)
            ye = euler(F.f5_1_c, t0, tend, n, y0)
            yrk = RK4(F.f5_1_c, t0, tend, n, y0)
            
            plot_results(x, y_exact, ye, yrk, 'Excercise 5.1 c)')
            
            print('Euler :', ye[-1], 'RK4 :', yrk[-1], 'Real :', y_exact[-1])
            
            print('d)')
            t0 = 0
            y0 = 1
            tend = 1
            n = 4
            
            x = np.linspace(t0,tend,n+1)
            y_exact = F.E5_1_d(x)
            ye = euler(F.f5_1_d, t0, tend, n, y0)
            yrk = RK4(F.f5_1_d, t0, tend, n, y0)
            
            plot_results(x, y_exact, ye, yrk, 'Excercise 5.1 d)')
            
            print('Euler :', ye[-1], 'RK4 :', yrk[-1], 'Real :', y_exact[-1])
        
        elif ans5 == 2:
            print('Excercise 5.2')
            t0 = 0
            tend = [1,2,3]
            y0 = 500
            
            table5 = Table()
            table5.add_column('años')
            table5.add_column('Trimestres')
            table5.add_column('Euler')
            table5.add_column('RK4')
            table5.add_column('Real Soltion')
            
            for i in range(0,3):
                n=(tend[i]*4)
                x = np.linspace(t0,tend[i],n+1)
                y_exact = F.E5_2(x)
                ye = euler(F.f5_2, t0, tend[i], n, y0)
                yrk = RK4(F.f5_2, t0, tend[i], n, y0)
                
                
                
                plot_results(x, y_exact, ye, yrk, 'Solution claculated')
                table5.add_row(str(tend[i]), str(n), str(ye[-1]), str(yrk[-1]), str(y_exact[-1]))
            
            console = Console()  
            console.print(table5)
        elif ans5 == 3:
            print('Excercise 5.3')
            t0 = 0
            y0 = 1/1000
            tend = 2/(y0)
            h = tend/2000
            n = int((tend-t0)/h)
    
            x = np.linspace(t0,tend,n+1)
            ye = euler(F.f5_3, t0, tend, n, y0)
            yrk = RK4(F.f5_3, t0, tend, n, y0)
            plot_results_non(x,ye,yrk, 'Excercise 5.3')
        elif ans5 == 4:
            print('Excercise 5.4')
            t0 = 1
            y0 = 1
            tend = 1.5
            h = [0.1,0.05]
            n1 = int((tend-t0)/h[0])
            n2 = int((tend-t0)/h[1])
            
            x = np.linspace(t0,tend,n1+1)
            y_exact = F.E5_4(x)
            ye = euler(F.f5_4, t0, tend, n1, y0)
            yrk = RK4(F.f5_4, t0, tend, n1, y0)
            
            plot_results(x, y_exact, ye, yrk, 'Excercise 5.4 h = 0.1')
            
            abs_error_euler = np.abs(y_exact-ye)
            rel_error_euler = abs_error_euler/y_exact
            
            abs_error_rk = np.abs(y_exact - yrk)
            rel_error_rk = abs_error_rk/y_exact
            
            Table5_4_e1 = Table()
            Table5_4_e1.add_column('xn')
            Table5_4_e1.add_column('euler solution')
            Table5_4_e1.add_column('Real Solution')
            Table5_4_e1.add_column('Absolute Error')
            Table5_4_e1.add_column('Relative Error')
            
            Table5_4_R1 = Table()
            Table5_4_R1.add_column('xn')
            Table5_4_R1.add_column('RK4 solution')
            Table5_4_R1.add_column('Real Solution')
            Table5_4_R1.add_column('Absolute Error')
            Table5_4_R1.add_column('Relative Error')
            
            for i in range(len(x)):
                Table5_4_e1.add_row(str(round(x[i],2)), str(ye[i]), str(y_exact[i]), str(round(abs_error_euler[i],16)), str(round(rel_error_euler[i],16)))
                Table5_4_R1.add_row(str(round(x[i],2)), str(yrk[i]), str(y_exact[i]), str(round(abs_error_rk[i],16)), str(round(rel_error_rk[i], 16)))
            
            console = Console()
            console.print(Table5_4_e1)
            console.print(Table5_4_R1)
            
            x = np.linspace(t0,tend,n2+1)
            y_exact = F.E5_4(x)
            ye = euler(F.f5_4, t0, tend, n2, y0)
            yrk = RK4(F.f5_4, t0, tend, n2, y0)
            
            
            plot_results(x, y_exact, ye, yrk, 'Excercise 5.4 h = 0.05')
            abs_error_euler = np.abs(y_exact-ye)
            rel_error_euler = abs_error_euler/y_exact
            
            abs_error_rk = np.abs(y_exact - yrk)
            rel_error_rk = abs_error_rk/y_exact
            
            Table5_4_e2 = Table()
            Table5_4_e2.add_column('xn')
            Table5_4_e2.add_column('euler solution')
            Table5_4_e2.add_column('Real Solution')
            Table5_4_e2.add_column('Absolute Error')
            Table5_4_e2.add_column('Relative Error')
            
            Table5_4_R2 = Table()
            Table5_4_R2.add_column('xn')
            Table5_4_R2.add_column('RK4 solution')
            Table5_4_R2.add_column('Real Solution')
            Table5_4_R2.add_column('Absolute Error')
            Table5_4_R2.add_column('Relative Error')
            
            for i in range(len(x)):
                Table5_4_e2.add_row(str(round(x[i],2)), str(ye[i]), str(y_exact[i]), str(round(abs_error_euler[i],16)), str(round(rel_error_euler[i],16)))
                Table5_4_R2.add_row(str(round(x[i],2)), str(yrk[i]), str(y_exact[i]), str(round(abs_error_rk[i],16)), str(round(rel_error_rk[i], 16)))
            
            console = Console()
            console.print(Table5_4_e2)
            console.print(Table5_4_R2)
        elif ans5 == 5:
            print('Excercise 5.5')
            t0 = 0
            y0 = 1
            tend = 10
            h = [0.25,0.1,0.05]
            n1 = int((tend-t0)/h[0])
            n2 = int((tend-t0)/h[1])
            n3 = int((tend-t0)/h[2])
            
            x = np.linspace(t0,tend,n1+1)
            ye = euler(F.f5_5, t0, tend, n1, y0)
            yrk = RK4(F.f5_5, t0, tend, n1, y0)
            
            plot_results_non(x, ye, yrk, 'Excercise 5.5, h = 0.25')
            
            x = np.linspace(t0,tend,n2+1)
            ye = euler(F.f5_5, t0, tend, n2, y0)
            yrk = RK4(F.f5_5, t0, tend, n2, y0)
            
            plot_results_non(x, ye, yrk, 'Excercise 5.5, h = 0.1')
            
            x = np.linspace(t0,tend,n3+1)
            ye = euler(F.f5_5, t0, tend, n3, y0)
            yrk = RK4(F.f5_5, t0, tend, n3, y0)
            
            plot_results_non(x, ye, yrk, 'Excercise 5.5, h = 0.05')
        
        elif ans5 == 6:
            ans5 = None
        else: 
            print('Not Valid option, Try Again')

def menu():
    """
    Function to display a menu for different practices and exercises.

    The user can choose a practice or exit the menu. Depending on the choice,
    the corresponding function (P1, P2, P3, P4, P5) is called to handle the exercises.

    Practices:
    1. Practice 1
    2. Practice 2
    3. Practice 3
    4. Practice 4
    5. Extra Exercise
    6. Exit

    Returns:
    - None
    """
    print("Project 3")
    ans = True
    while ans:
        print("1.Practice 1")
        print("2.Practice 2")
        print("3.Practice 3")
        print("4.Practice 4")
        print("5.Extra Excercise")
        print("6.Exit")
        ans = int(input("please select what would you like to do"))
        if ans == 1:
            P1()
        elif ans == 2:
            P2()
        elif ans == 3:
            P3()
        elif ans == 4:
            P4()
        elif ans == 5:
            P5()
        elif ans == 6:
            return None
        else:
            print("\n Not Valid Choose Try Again")

            
menu()