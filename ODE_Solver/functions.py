import numpy as np
"""
Practice 1:
1. f1_1(x, y): y' = x - y
   E1_1(x): y = x + 5 * exp(-x) - 1

2. f1_2(x, y): y' = 2 - y / x (with handling for x = 0)
   E1_2(x): y = x - 1 / x

3. f1_3(x, y): y' = (x^6 * e^x + 4y) / x (with handling for x = 0)
   E1_3(x): y = e^x * (x^5 - x^4)

4. f1_4(x, y): y' = -0.00012378 * y
   E1_4(x): y = 100 * exp(-0.00012378 * x)

5. f1_5(x, y): y' = -0.19018 * (y - 70)
   E1_5(x): y = 230 * exp(-0.19018 * x) + 70

6. f1_6(x, y): y' = 6 - (1/100) * y
   E1_6(x): y = 600 - 550 * exp(-x / 100)

Practice 2:
1. f2_1(t, P): P' = 0.04 * P
   E2_1(x): y = 5000 * exp(0.04 * x)

2. f2_2(t, P, r): P' = r * P
   E2_2(x, r): y = 5000 * exp(x * r)

3. f2_3(t, P): P' = 0.03 * P
   E2_3(x): y = 10000 * exp(0.03 * x)

Practice 3:
1. f3_1(x, y): y' = y^2 - y^3

Practice 4:
1. f4_1(x, y): y' = y
   E4_1(x): y = exp(x)

2. f4_2(x, y): y' = 2 * cos(x) * y
   E4_2(x): y = exp(2 * sin(x))

Practice 5:
1. f5_1_a(x, y): y' = x * exp(3 * x) - 2 * y
   E5_1_a(x): y = (1/5) * x * exp(3 * x) - (1/25) * exp(3 * x) + (1/25) * exp(-2 * x)

2. f5_1_b(x, y): y' = 1 + (x - y)^2
   E5_1_b(x): y = x - 1 / (1 - x)

3. f5_1_c(x, y): y' = 1 + y / x (with handling for x = 0)
   E5_1_c(x): y = x * log(x) + 2 * x

4. f5_1_d(x, y): y' = cos(2 * x) + sin(3 * x)
   E5_1_d(x): y = (1/2) * sin(2 * x) - (1/3) * cos(3 * x) + 4 / 3

5. f5_2(x, y): y' = 0.0375 * y
   E5_2(x): y = 500 * exp(0.0375 * x)

6. f5_3(x, y): y' = y^2 - y^3

7. f5_4(x, y): y' = 2 * x * y
   E5_4(x): y = exp(x^2 - 1)
"""

def f1_1(x,y): # parte derecha de la edo y'=f(x,y)
    z = x - y
    return z

def E1_1(x): # solucion analitica (funcion matematico)
    y = x + 5*np.exp(-x) - 1
    return y

def f1_2(x,y): # parte derecha de la edo y'=f(x,y)
    if x == 0:  # Manejo de división por cero
        z = float(0)  # Podría ser otra solución dependiendo del contexto
    else:
        z = 2 - y / x
    return z

def E1_2(x): # solucion analitica (funcion matematico)
    y = x-(1/x)
    return y

def f1_3(x,y): # parte derecha de la edo y'=f(x,y)
    if x == 0:  # Manejo de división por cero
        z = float(0)  # Podría ser otra solución dependiendo del contexto
    else:
        z = (np.power(x,6)*np.power(np.e,x)+4*y)/x
    return z

def E1_3(x): # solucion analitica (funcion matematico)
    y = np.exp(x)*(np.power(x,5)-np.power(x,4))
    return y

def f1_4(x,y): # parte derecha de la edo y'=f(x,y)
    z = -0.00012378*y
    return z

def E1_4(x): # solucion analitica (funcion matematico)
    y = 100*np.exp(-0.00012378*x)
    return y

def f1_5(x,y): # parte derecha de la edo y'=f(x,y)
    z = -0.19018*(y-70)
    return z

def E1_5(x): # solucion analitica (funcion matematico)
    y = 230*np.exp(-0.19018*x)+70
    return y

def f1_6(x,y): # parte derecha de la edo y'=f(x,y)
    z = 6-((1/100)*y)
    return z

def E1_6(x): # solucion analitica (funcion matematico)
    y = 600-550*np.exp(-x/100)
    return y

def f2_1(t,P):
    z = 0.04*P
    return z    

def E2_1(x): # solucion analitica (funcion matematico)
    y = 5000*np.exp(0.04*x)
    return y

def f2_2(t,P,r):
    z = r*P 
    return z

def E2_2(x,r):
    y = 5000*np.exp(x*r)
    return y

def f2_3(t,P):
    z = 0.03*P
    return z

def E2_3(x):
    y = 10000*np.exp(0.03*x)
    return y

def f3_1(x,y):
    z=np.power(y,2)-np.power(y,3)
    return z

def f4_1(x,y):
    z = y
    return z

def E4_1(x):
    y = np.exp(x)
    return y

def f4_2(x,y):
    z = 2*np.cos(x)*y
    return z

def E4_2(x):
    y = np.exp(2*np.sin(x))
    return y

def f5_1_a(x,y):
    z = x*np.exp(3*x)-2*y
    return z

def E5_1_a(x):
    y = (1/5)*x*np.exp(3*x)-(1/25)*np.exp(3*x) + (1/25)*np.exp(-2*x)
    return y

def f5_1_b(x,y):
    z = 1 + np.power(x-y,2)
    return z

def E5_1_b(x):
    y = x+(1/(1-x))
    return y

def f5_1_c(x,y):
    if x == 0:
        return 0
    else:
        z = 1 +(y/x)
        return z
    
def E5_1_c(x):
    y = x * np.log(x) + 2 * x
    return y

def f5_1_d(x,y):
    z = np.cos(2*x)+np.sin(3*x)
    return z 

def E5_1_d(x):
    y = ((1/2)*np.sin(2*x))-(1/3)*np.cos(3*x)+(4/3)
    return y

def f5_2(x,y):
    z = 0.0375*y
    return z

def E5_2(x):
    y = 500*np.exp(0.0375*x)
    return y

def f5_3(x,y):
    z = np.power(y,2)-np.power(y,3)
    return z

def f5_4(x,y):
    z = 2*(x*y)
    return z

def E5_4(x):
    y = np.power(np.e, np.power(x,2)-1)
    return y

def f5_5(x,y):
    z = y*(10-2*y)
    return z