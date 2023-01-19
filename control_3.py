#Tercer script donde resolvere un modelos de step-space
import scipy.signal as sig
import numpy as np
import matplotlib.pyplot as plt

#Simulation parameters
x0 = np.array([0,0])
start = 0
stop = 30
step = 1
t = np.arange(start,stop,step)

K = 3
T = 4

#Satate-space Model
A = np.array([[-1/T, 0], [0, 0]])
B = np.array([[K/T], [0]])
C = np.array([[1,0]])
D = np.array([0])

sys = sig.StateSpace(A,B,C,D)

#Step Response

t, y = sig.step(sys, x0, t)

#Plotting
plt.plot(t,y)
plt.title('Step Response')
plt.xlabel('t')
plt.ylabel('y')
plt.grid()
plt.show()