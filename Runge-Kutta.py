import numpy as np
import sympy as sp
from math import *

def crear_funcion(expr_str, variables):
    """Convierte una cadena de texto en una función evaluable"""
    # Reemplazar '^' por '**' para potencias
    expr_str = expr_str.replace('^', '**')
    
    # Crear una función lambda que evalúe la expresión
    var_names = ','.join(variables)
    func_str = f"lambda {var_names}: {expr_str}"
    return eval(func_str)

def rk4_first_order(f, t0, y0, tf, h):
    """Método Runge-Kutta de cuarto orden para EDO de primer orden"""
    t = np.arange(t0, tf + h, h)
    n = len(t)
    y = np.zeros(n)
    y[0] = y0
    
    for i in range(n-1):
        k1 = f(t[i], y[i])
        k2 = f(t[i] + h/2, y[i] + h*k1/2)
        k3 = f(t[i] + h/2, y[i] + h*k2/2)
        k4 = f(t[i] + h, y[i] + h*k3)
        y[i+1] = y[i] + h*(k1 + 2*k2 + 2*k3 + k4)/6
    
    return y[-1]

def rk4_second_order(f1, f2, t0, y0, dy0, tf, h):
    """Método Runge-Kutta de cuarto orden para EDO de segundo orden"""
    t = np.arange(t0, tf + h, h)
    n = len(t)
    y = np.zeros(n)
    dy = np.zeros(n)
    y[0] = y0
    dy[0] = dy0
    
    for i in range(n-1):
        k1_y = dy[i]
        k1_dy = f2(t[i], y[i], dy[i])
        
        k2_y = dy[i] + h*k1_dy/2
        k2_dy = f2(t[i] + h/2, y[i] + h*k1_y/2, dy[i] + h*k1_dy/2)
        
        k3_y = dy[i] + h*k2_dy/2
        k3_dy = f2(t[i] + h/2, y[i] + h*k2_y/2, dy[i] + h*k2_dy/2)
        
        k4_y = dy[i] + h*k3_dy
        k4_dy = f2(t[i] + h, y[i] + h*k3_y, dy[i] + h*k3_dy)
        
        y[i+1] = y[i] + h*(k1_y + 2*k2_y + 2*k3_y + k4_y)/6
        dy[i+1] = dy[i] + h*(k1_dy + 2*k2_dy + 2*k3_dy + k4_dy)/6
    
    return y[-1], dy[-1]

def rk4_system(f1, f2, t0, x0, y0, tf, h):
    """Método Runge-Kutta de cuarto orden para sistemas 2x2"""
    t = np.arange(t0, tf + h, h)
    n = len(t)
    x = np.zeros(n)
    y = np.zeros(n)
    x[0] = x0
    y[0] = y0
    
    for i in range(n-1):
        k1_x = f1(t[i], x[i], y[i])
        k1_y = f2(t[i], x[i], y[i])
        
        k2_x = f1(t[i] + h/2, x[i] + h*k1_x/2, y[i] + h*k1_y/2)
        k2_y = f2(t[i] + h/2, x[i] + h*k1_x/2, y[i] + h*k1_y/2)
        
        k3_x = f1(t[i] + h/2, x[i] + h*k2_x/2, y[i] + h*k2_y/2)
        k3_y = f2(t[i] + h/2, x[i] + h*k2_x/2, y[i] + h*k2_y/2)
        
        k4_x = f1(t[i] + h, x[i] + h*k3_x, y[i] + h*k3_y)
        k4_y = f2(t[i] + h, x[i] + h*k3_x, y[i] + h*k3_y)
        
        x[i+1] = x[i] + h*(k1_x + 2*k2_x + 2*k3_x + k4_x)/6
        y[i+1] = y[i] + h*(k1_y + 2*k2_y + 2*k3_y + k4_y)/6
    
    return x[-1], y[-1]

def main():
    print("\nInstrucciones para ingresar funciones:")
    print("- Use 't' para la variable independiente")
    print("- Use 'y' para la función (primer orden)")
    print("- Use 'y' y 'dy' para la función y su derivada (segundo orden)")
    print("- Use 'x' y 'y' para sistemas de ecuaciones")
    print("- Funciones disponibles: sin, cos, tan, exp, log, sqrt")
    print("- Use '^' o '**' para potencias")
    print("Ejemplos:")
    print("- t + y")
    print("- sin(t*y)")
    print("- exp(t*y)")
    print("- t^2 * y")

    while True:
        print("\nQué desea resolver?")
        print("1. Ecuación diferencial de primer orden")
        print("2. Ecuación diferencial de segundo orden")
        print("3. Sistemas 2x2 de ecuaciones diferenciales")
        print("4. Salir")
        
        opcion = input("\nSeleccione una opción: ")
        
        if opcion == '1':
            print("\nPara una ecuación de la forma y'(t) = f(t,y)")
            expr = input("Ingrese la función f(t,y): ")
            t0 = float(input("Ingrese t0: "))
            y0 = float(input("Ingrese y0: "))
            tf = float(input("Ingrese tf: "))
            h = float(input("Ingrese h: "))
            
            try:
                f = crear_funcion(expr, ['t', 'y'])
                resultado = rk4_first_order(f, t0, y0, tf, h)
                print(f"\ny({tf}) ≈ {resultado:.6f}")
            except Exception as e:
                print(f"Error al evaluar la función: {e}")
            
        elif opcion == '2':
            print("\nPara una ecuación de la forma y''(t) = f(t,y,y')")
            expr = input("Ingrese la función f(t,y,y'): ")
            t0 = float(input("Ingrese t0: "))
            y0 = float(input("Ingrese y0: "))
            dy0 = float(input("Ingrese y'0: "))
            tf = float(input("Ingrese tf: "))
            h = float(input("Ingrese h: "))
            
            try:
                f1 = lambda t, y, dy: dy
                f2 = crear_funcion(expr, ['t', 'y', 'dy'])
                
                y_final, dy_final = rk4_second_order(f1, f2, t0, y0, dy0, tf, h)
                print(f"\ny({tf}) ≈ {y_final:.6f}")
                print(f"y'({tf}) ≈ {dy_final:.6f}")
            except Exception as e:
                print(f"Error al evaluar la función: {e}")
            
        elif opcion == '3':
            print("\nPara un sistema de la forma:")
            print("x'(t) = f1(t,x,y)")
            print("y'(t) = f2(t,x,y)")
            expr1 = input("Ingrese la función f1(t,x,y): ")
            expr2 = input("Ingrese la función f2(t,x,y): ")
            t0 = float(input("Ingrese t0: "))
            x0 = float(input("Ingrese x0: "))
            y0 = float(input("Ingrese y0: "))
            tf = float(input("Ingrese tf: "))
            h = float(input("Ingrese h: "))
            
            try:
                f1 = crear_funcion(expr1, ['t', 'x', 'y'])
                f2 = crear_funcion(expr2, ['t', 'x', 'y'])
                
                x_final, y_final = rk4_system(f1, f2, t0, x0, y0, tf, h)
                print(f"\nx({tf}) ≈ {x_final:.6f}")
                print(f"y({tf}) ≈ {y_final:.6f}")
            except Exception as e:
                print(f"Error al evaluar la función: {e}")
            
        elif opcion == '4':
            print("\n¡Hasta luego!")
            break
        
        else:
            print("\nOpción no válida. Por favor, seleccione una opción del 1 al 4.")

if __name__ == "__main__":
    main()