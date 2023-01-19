#Quinto script de resolucion de problemas de control
#Resolviendo la trasformada anterior con la libreria control
import numpy as np
import control
import matplotlib.pyplot as plt

#Define Transfer Funtion  H(s)=3(2s+1)/(3s+1)(5s+1)
num1 = np.array([3])
num2 = np.array([2,1])
num = np.convolve(num1, num2)

den1  = np.array([3,1])
den2  = np.array([5,1])
den  = np.convolve(den1, den2)

H = control.tf(num, den)
print('H(s) = ', H)

#Bode plot
#control.bode_plot(H)

#Bode plot
w, mag, phase = control.bode(H, dB=True)

#print(w.shape)
#print('*'*80)
#print(mag.shape)
#print('*'*80)
#print(phase.shape)
#print('*'*80)

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