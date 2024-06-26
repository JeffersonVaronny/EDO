"""Este módulo permite al usuario resolver Ecuaciones Diferenciales Ordinarias con hasta tres metodos numericos. 

El módulo contiene las siguientes funciones:

- `Euler (f, t, x0)` - Retorna una lista con los valores de x calculados usando el Método de Euler, para cada valor t dado.
- `RK2 (f, t, x0)` - Retorna una lista con los valores de x calculados usando el Método de Runge-Kutta 2º Orden, para cada valor t dado.
- `RK4 (f, t, x0)` - Retorna una lista con los valores de x calculados usando el Método de Runge-Kutta 4º Orden, para cada valor t dado.

Los ejmplos mostrados son para una EDO normalizada de la forma 

    dx/dt = -x^3 + sin(t)



"""
import numpy as np
import matplotlib.pyplot as plt


def Euler (f: 'function', t: 'list', x0: float)-> 'list':
    """Resuelve una Ecuaciones Diferenciales Ordinarias usando el Método de Euler.

    Examples:
    
        import numpy as np
        import matplotlib.pyplot as plt
        
        def fuc (x,t):
            return -(x*x*x) + np.sin(t)
        
        t=np.linspace(0.0, 10, 20)
        x=Euler(fuc, t20, 0)
        plt.plot(t, x)
        plt.show()

    Args:
        f: Funcion F(x,t).
        t: Areglo,separado equidistante, que contiene los t donde se quiere realizar el calculo.
        x0: Condicion inicial x(t_0)=x_0.

    Returns:
        x: retorna una arreglo ordenado con los valores resultantes x(t) para cada t.

    """ 
    h=t[1] - t[0]
    x=np.zeros(t.size)
    x[0]=x0
    for i in range (t.size-1):
        x[i+1] = x[i] + h * f(x[i], t[i])
         
    return x

def RK2 (f: 'function', t: 'list', x0: float)-> 'list':
    """Resuelve una Ecuaciones Diferenciales Ordinarias usando el Método de Runge-Kutta 2º Orden.

    Examples:
    
        import numpy as np
        import matplotlib.pyplot as plt
        
        def fuc (x,t):
            return -(x*x*x) + np.sin(t)
        
        t=np.linspace(0.0, 10, 20)
        x=RK2(fuc, t20, 0)
        plt.plot(t, x)
        plt.show()

    Args:
        f: Funcion F(x,t).
        t: Areglo,separado equidistante, que contiene los t donde se quiere realizar el calculo.
        x0: Condicion inicial x(t_0)=x_0.

    Returns:
        x: retorna una arreglo ordenado con los valores resultantes x(t) para cada t.

    """  
    h=t[1] - t[0]
    x=np.zeros(t.size)
    x[0]=x0
    for i in range (t.size-1):
        k1 = h * f(x[i], t[i])
        k2 = h * f(x[i] + k1/2, t[i] + h/2)
        x[i+1] = x[i] + k2
         
    return x

def RK4 (f: 'function', t: 'list', x0: float)-> 'list':
    """Resuelve una Ecuaciones Diferenciales Ordinarias usando el Método de Runge-Kutta 4º Orden.

    Examples:
    
        import numpy as np
        import matplotlib.pyplot as plt
        
        def fuc (x,t):
            return -(x*x*x) + np.sin(t)
        
        t=np.linspace(0.0, 10, 20)
        x=RK4(fuc, t20, 0)
        plt.plot(t, x)
        plt.show()

    Args:
        f: Funcion F(x,t).
        t: Areglo,separado equidistante, que contiene los t donde se quiere realizar el calculo.
        x0: Condicion inicial x(t_0)=x_0.

    Returns:
        x: retorna una arreglo ordenado con los valores resultantes x(t) para cada t.

    """ 
    h=t[1] - t[0]
    x=np.zeros(t.size)
    x[0]=x0
    for i in range (t.size-1):
        k1 = h * f(x[i], t[i])
        k2 = h * f(x[i] + k1/2, t[i] + h/2)
        k3 = h * f(x[i] + k2/2, t[i] + h/2)
        k4 = h * f(x[i] + k3, t[i] + h)
        x[i+1] = x[i] + (1/6) * (k1 +2*k2 + 2*k3 +k4)
         
    return x
    
def main():
    
    def fuc (x: float,t: float)-> float:
        return -(x*x*x) + np.sin(t)
    
    t20=np.linspace(0.0, 10, 20)
    tmil=np.linspace(0.0, 10, 1000)
    x20=Euler(fuc, t20, 0)
    xmil=Euler(fuc, tmil, 0)
    
    plt.plot(t20, x20)
    plt.plot(tmil, xmil)
    plt.show()
    
    x20_RK2=RK2(fuc, t20, 0)
    xmil_RK2=RK2(fuc, tmil, 0)
    plt.plot(t20, x20_RK2)
    plt.plot(tmil, xmil_RK2)
    plt.show()
    
    x20_RK4=RK4(fuc, t20, 0)
    xmil_RK4=RK4(fuc, tmil, 0)
    plt.plot(t20, x20_RK4)
    plt.plot(tmil, xmil_RK4)
    plt.show()

if __name__=="__main__":
    main()