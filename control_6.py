#sexto Script de control con python
#Resolviendo un sistema de PI de primer orden

import numpy as np
import matplotlib.pyplot as plt

#Datos del modelo
K = 3
T = 4
a =-(1/T)
b = K/T

#Parametros de la simualcion
Ts = 0.1 #Sampling time
Tstop = 20 #End of the simulation
N = int(Tstop/Ts) #Simulation legth
y = np.zeros(N+2) #Initializacion of the vector Thau
y[0] = 0 #Initial Value value

#PI controller settings
Kp = 0.5 
Ti = 5

r = 5
e = np.zeros(N+2)
u = np.zeros(N+2)

#Simulation
for k in range(N+1):
    e[k] = r-y[k]
    u[k] = u[k-1]+Kp*(e[k]-e[k-1]) + (Kp/Ti)*e[k]
    y[k+1] = (1+Ts*a)*y[k] + Ts*b*u[k]

t = np.arange(0, Tstop+2*Ts, Ts)

#Grafica de los valores resultantes
plt.figure(1)
plt.plot([Ts,Tstop],[r,r], color='red', label = 'Reference')
plt.plot(t,y, label='Signal')
plt.grid()
plt.legend()
plt.title('Control Dynamic System')
plt.xlabel('t[s]')
plt.ylabel('y')


plt.figure(2)
plt.plot(t,u, label = 'Control Signal')
plt.legend()
plt.grid()
plt.title('Control Signal')
plt.xlabel('t[s]')
plt.ylabel('u[y]')
plt.show()