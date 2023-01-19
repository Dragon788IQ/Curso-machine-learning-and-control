#Septimo script de control
#Resolviendo un ejemplo de estabilidad
import numpy as np
import control 
import matplotlib.pyplot as plt

#Transfer Function Process
K = 3
T = 4
num_p = np.array([K])
den_p = np.array([T, 1])
Hp = control.tf(num_p, den_p)
print('Hp(s) = ', Hp)

#Transfer Function PI controller
Kp = 0.4
Ti = 2
num_c = np.array([Kp*Ti, Kp])
den_c = np.array([Ti, 0])
Hc = control.tf(num_c, den_c)
print('Hc(s) = ', Hc)

#Transfer Function Mesurement
Tm = 1
num_m = np.array([1])
den_m = np.array([Tm, 1])
Hm = control.tf(num_m, den_m)
print('Hm(s) = ', Hm)

#Transfer Function Lowpass Filter
Tf = 1
num_f = np.array([1])
den_f = np.array([Tf, 1])
Hf = control.tf(num_f, den_f)
print('Hm(s) = ', Hf)

#The Loop Transfer function
L = control.series(Hc, Hp, Hf, Hm)
print('L(s) = ', L)

#Tracking transfer Function
T = control.feedback(L,1)
print('T(s) = ', T)

#Step Response Feedback System (Tracking system)
t,y = control.step_response(T)
plt.figure(1)
plt.plot(t,y)
plt.title('Step Response Feedback System')
plt.grid()

#Bode Diagram with Stability Margins
plt.figure(2)
control.bode(L, dB=True, deg=True, margins=True)

#Poles and Zeros
plt.figure(3)
control.pzmap(T)
p = control.pole(T)
z = control.zero(T)
print('Poles = ', p)

#Calculating stability margins and crossover frequencies
gm, pm, w180, wc = control.margin(L)

#Conver gm to decibels
gmdb = 20 * np.log10(gm)

print(f"wc = {wc} rad/s")
print(f"w180 = {w180} rad/s")

print(f"GM = {gm}")
print(f"GM = {gmdb} dB")
print(f"PM = {pm} deg")

#Find when the system is marginally stable (Kritical Gan - Kc)
kc = Kp*gm
print(f"Kc = {kc}")
plt.show()