#Segundo script para la solucion de problemas de control
#Resolviendo transformadas de primer orden
import control
import numpy as np
import matplotlib.pyplot as plt

#Resolviendo la ecuacion de la forma:
# H(s) = K/Ts+1 = 3/4s+1

num = np.array([3])
den = np.array([4,1])

#Inicalizando la ecuacion de la funcion H(s)
H = control.tf(num, den)
print('H(s)= ', H)

#Resolviendo el sistema
t,y = control.step_response(H)

#ploteando los resultados
plt.plot(t,y)
plt.title('Step Response')
plt.grid()
plt.show()