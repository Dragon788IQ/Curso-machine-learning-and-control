#Primer script de practica para resolucion de problemas de control
#Resolviendo ODES de primer orden como si fuera ode45
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

#Inicializacion de los paremetros a usar
K = 3
T = 4
u = 1

tstart = 0
tstop = 25
increment = 0.1
y0 = 0
t  = np.arange(tstart, tstop, increment)

#Funtion ra returns the dx/dt
def system1orden (y, t, T, K, u):
    dydt = (1/T)*(-y + K*u)
    return dydt

#Solve the ODE
x = odeint(system1orden, y0, t, args=(K, T, u))
print(x)

#Plot the result
plt.plot(t,x)
plt.title("1.Orden System dyct=(1/T)(-y + K*u)")
plt.xlabel("t")
plt.ylabel("y(t)")
plt.show()
