import numpy as np
import matplotlib.pyplot as plt

# Чтение данных из файла
data = np.loadtxt('data/data_20260116_162855.txt')

# Извлечение второго столбца (время) и третьего столбца (A0)
time = data[:, 1]  # Время эксперимента
A0_raw = data[:, 2]  # Напряжение A0

# Преобразование времени (если нужно нормализовать, начинаем с 0)
time_normalized = time - time[0]

# Определяем, в каких единицах A0 (похоже на цифровые значения 0-1023)
# Если значения больше 5, то это цифровые значения
# Если меньше 5, то это вольты

# Предполагаем, что A0 уже в цифровых единицах (0-1023)
# Если нужно конвертировать из вольт: A0_digital = (A0_volts / 5.0) * 1023
# Если уже цифровые: используем как есть

A0_digital = A0_raw  # Предполагаем, что это уже цифровые значения

# Если значения выглядят как вольты (меньше 5), конвертируем
if np.max(A0_raw) < 10:
    A0_digital = (A0_raw / 5.0) * 1023

# Конвертация в вольты для расчетов сопротивления и тока
A0_volts = (A0_digital / 1023.0) * 5.0

# Расчет сопротивления M из уравнения: 5/(220+M) = A0/M
# 5M = A0(220+M)
# 5M = 220*A0 + A0*M
# 5M - A0*M = 220*A0
# M(5 - A0) = 220*A0
# M = 220*A0/(5 - A0)
# Фильтруем значения, где A0 < 5 (иначе деление на ноль или отрицательные значения)
valid_mask = A0_volts < 5.0
resistance = np.zeros_like(A0_volts)
resistance[valid_mask] = 220.0 * A0_volts[valid_mask] / (5.0 - A0_volts[valid_mask])
resistance[~valid_mask] = np.nan

# Расчет силы тока: I = A0/M
current = np.zeros_like(A0_volts)
# Дополнительная проверка: сопротивление не должно быть нулевым или очень малым
valid_current_mask = valid_mask & (resistance > 1e-6)
current[valid_current_mask] = A0_volts[valid_current_mask] / resistance[valid_current_mask]
current[~valid_current_mask] = np.nan

# Создание графиков
fig, axes = plt.subplots(3, 1, figsize=(12, 10))

# График 1: Зависимость времени от цифровых значений напряжения
axes[0].plot(time_normalized, A0_digital, 'b-', linewidth=0.5)
axes[0].set_xlabel('Время (мс)', fontsize=12)
axes[0].set_ylabel('Цифровое значение напряжения', fontsize=12)
axes[0].set_title('График 1: Зависимость цифрового значения напряжения от времени', fontsize=14)
axes[0].grid(True, alpha=0.3)

# График 2: Зависимость сопротивления от времени
axes[1].plot(time_normalized[valid_mask], resistance[valid_mask], 'r-', linewidth=0.5)
axes[1].set_xlabel('Время (мс)', fontsize=12)
axes[1].set_ylabel('Сопротивление M (Ом)', fontsize=12)
axes[1].set_title('График 2: Зависимость сопротивления от времени', fontsize=14)
axes[1].grid(True, alpha=0.3)

# График 3: Зависимость силы тока от времени
axes[2].plot(time_normalized[valid_current_mask], current[valid_current_mask] * 1000, 'g-', linewidth=0.5)  # Умножаем на 1000 для мА
axes[2].set_xlabel('Время (мс)', fontsize=12)
axes[2].set_ylabel('Сила тока (мА)', fontsize=12)
axes[2].set_title('График 3: Зависимость силы тока от времени', fontsize=14)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('data/graphs.png', dpi=300, bbox_inches='tight')
print("Графики сохранены в файл data/graphs.png")
plt.show()

#%%
# Находим минимальное и максимальное сопротивление
Rmin = resistance[np.where((time_normalized > 200000) & (time_normalized < 250000))[0]].min()
Rmax = resistance[np.where((time_normalized > 0) & (time_normalized < 100000))[0]].max()

print("Rmin = ", Rmin)
print("Rmax = ", Rmax)


# Выделяем участок зависимости для обработки
t_start = 210000
t_stop = 300000

ind_crop = np.where((time_normalized > t_start) & (time_normalized < t_stop))[0]
time_crop = time_normalized[ind_crop]
current_crop = current[ind_crop]
resistance_crop = resistance[ind_crop]

fig, axes = plt.subplots(2, 1, figsize=(12, 10))
axes[0].plot(time_crop, resistance_crop)
axes[0].set_title("R")
axes[1].plot(time_crop, current_crop)
axes[1].set_title("I")
plt.show()

# Сглаживаем графики
from scipy.ndimage import uniform_filter1d
window_size = 100
time_av = uniform_filter1d(time_crop, size=window_size)
current_av = uniform_filter1d(current_crop, size=window_size)
resistance_av = uniform_filter1d(resistance_crop, size=window_size)

fig, axes = plt.subplots(2, 1, figsize=(12, 10))
axes[0].plot(time_crop, resistance_crop)
axes[0].plot(time_av, resistance_av)
axes[0].set_title("R")
axes[1].plot(time_crop, current_crop)
axes[1].plot(time_av, current_av)
axes[1].set_title("I")
plt.show()

# Определеяем состояние резистора X и его производную
X = (resistance_av - Rmax) / (Rmin - Rmax)
dXdt = np.gradient(X, time_crop)
fig, axes = plt.subplots(2, 1, figsize=(12, 10))
axes[0].plot(time_av, X)
axes[0].set_title("X")
axes[1].plot(time_av, dXdt)
axes[1].set_title("dX/dt")
plt.show()

# Рассчитываем функцию скорости изменения состояния
v = dXdt / current_av
plt.plot(X, v)
plt.show()

# Аппроксимируем
from scipy.optimize import curve_fit

# Ручная функция параболы
# def parabola(x, a1, a2, x1, x2, y0):
#     return a1 * (x - x1)**2 + a2 * (x - x2)**2 + y0

# ind_fit = np.where((X > 0.25) & (X < 0.8))[0]
# p0=[0.3, 0.3, 0.42, 0.7, -0.007]
# popt, pcov = curve_fit(parabola, X[ind_fit], v[ind_fit], p0=p0)
# v_fit = parabola(X, *popt)

def gauss(x, a1, a2, x1, x2, w1, w2, y0):
    return a1 * np.exp(-((x - x1) / w1)**2) + a2 * np.exp(-((x - x2) / w2)**2) + y0

p0=[-0.1, -0.1, 0.42, 0.7, 0.05, 0.05, 0.007]
popt, pcov = curve_fit(gauss, X, v, p0=p0)
v_fit = gauss(X, *popt)



print("Параметры:", popt)

# График сравнения
plt.figure(figsize=(8, 6))
plt.plot(X, v, label='Исходные данные')
plt.plot(X, v_fit, label='Фит', color='red')
# plt.plot(X, parabola(X, *p0), 'b-', label='Фит параболой')
plt.legend()
plt.xlabel('X')
plt.ylabel('dX/dt / I')
plt.grid(True, alpha=0.3)
plt.show()





