#Cuarto Script de resolucion de problemas de control
#Resolucion de una transformada obteniendo la magnitud y la fase
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

#Define Trasfer Funtion H(s)=3(2s+1)/(3s+1)(5s+1)
num1 = np.array([3])
num2 = np.array([2,1])
num = np.convolve(num1,num2)

den1 = np.array([3,1])
den2 = np.array([5,1])
den = np.convolve(den1, den2) 

H = signal.TransferFunction(num, den)
print('H(s) = ', H)

#Frecuencies
w_start = 0.01 
w_stop = 10
step = 0.01
N = int((w_stop-w_start)/step)+1
w = np.linspace(w_start, w_stop, N)

w, mag, phase = signal.bode(H,w)
#Bode plot

plt.figure()
plt.subplot(2,1,1)
plt.semilogx(w,mag) #Bode magnitude plot
plt.title('Bode plot')
plt.ylabel('Magnitude (dB)')
plt.grid(b=None, which='major', axis='both')
plt.grid(b=None, which='minor', axis='both')

plt.subplot(2,1,2)
plt.semilogx(w,phase) #Bode Phase plot
plt.ylabel('Phase (deg)')
plt.xlabel('Frecuency (rad/seg)')
plt.grid(b=None, which='major', axis='both')
plt.grid(b=None, which='minor', axis='both')
plt.show()