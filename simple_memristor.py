#!/usr/bin/env python
# coding: utf-8

# In[100]:


import numpy as np
import matplotlib.pyplot as plt


# In[101]:


class Memristor():
    def __init__(self, x_start=0.5, R_on=100.0, R_off=1600.0, k=1e5,p=2):
        """
        модель Вудса-Уиллиса
        - x: текущее состояние (0..1)
        - R_on, R_off: минимальное и максимальное сопротивление (Ом)
        - k: коэффициент скорости изменения состояния
        """
        self.x = x_start
        self.R_on = R_on
        self.R_off = R_off
        self.k = k
        self.p = 1

    def get_R(self):
        return self.R_off + (self.R_on - self.R_off) * self.x

    def get_x(self):
        return self.x * 1.

    def upd_x(self, dq):
        self.x += self.k * dq * (1 - (2*self.x - 1)**self.p)
        self.x = max(min(0.99, self.x), 0.01)


# In[102]:


# ----------------------------
# Параметры схемы и моделирования
# ----------------------------
# Начальное состояние мемристора
mem = Memristor() # мемристор
Rr = 1000.0         # последовательный резистор, Ом
T = 1                # время моделирования, с
dt = 1e-3                # шаг по времени, с
N = int(T/dt)

t = np.linspace(0, T, N)

# Источник переменного напряжения
f = 4
V_src = np.sin(2 * np.pi * f * t)

# Массивы для результатов
x_arr = np.zeros(N)
I_arr = np.zeros(N)
Vm_arr = np.zeros(N)


# In[103]:


# ----------------------------
# Цикл вычислений
# ----------------------------
for i in range(N):
    Rm = mem.get_R()
    I = V_src[i] / (Rm + Rr)
    Vm = I * Rm

    x_arr[i] = mem.get_x()
    I_arr[i] = I
    Vm_arr[i] = Vm

    mem.upd_x(I * dt)


# In[104]:


# ----------------------------
# Вывод графиков
# ----------------------------
plt.figure(figsize=(10, 16))

plt.subplot(4, 1, 1)
plt.plot(t, V_src, label='Источник V(t)')
plt.plot(t, Vm_arr, label='Напряжение на мемристоре Vm(t)')
plt.ylabel('Вольты, В')
plt.legend()
plt.grid(True)

plt.subplot(4, 1, 2)
plt.plot(t, I_arr)
plt.ylabel('Ток, А')
plt.title('Ток через цепь')
plt.grid(True)

plt.subplot(4, 1, 3)
plt.plot(t, x_arr)
plt.xlabel('Время, с')
plt.ylabel('Состояние x(t)')
plt.title('Динамика состояния мемристора')
plt.grid(True)

plt.subplot(4, 1, 4)  
plt.plot(Vm_arr, I_arr)
plt.xlabel('Вольты, В')
plt.ylabel('Ток, А')
plt.title('Динамика состояния мемристора')
plt.grid(True)

plt.tight_layout()
plt.show()


# In[ ]:





# In[ ]:




