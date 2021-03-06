# Задание 1
# Загрузите модуль pyplot библиотеки matplotlib с псевдонимом plt,
# а также библиотеку numpy с псевдонимом np.
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
%config InlineBackend.figure_format='svg'
# Примените магическую функцию %matplotlib inline для
# отображения графиков в Jupyter Notebook и
# настройки конфигурации ноутбука со значением 'svg' для
# более четкого отображения графиков.


#%%
x = [1, 2, 3, 4, 5, 6, 7]
y = [3.5, 3.8, 4.2, 4.5, 5, 5.5, 7]

plt.plot(x,y)
# Создайте список под названием x с числами 1, 2, 3, 4, 5, 6, 7
# и список y с числами 3.5, 3.8, 4.2, 4.5, 5, 5.5, 7.
# С помощью функции plot постройте график, соединяющий линиями точки 
# с горизонтальными координатами из списка x
# и вертикальными - из списка y.

# Затем в следующей ячейке постройте диаграмму рассеяния
# (другие названия - диаграмма разброса, scatter plot).
#%%
plt.scatter(x, y)


# Задание 2
# С помощью функции linspace из библиотеки Numpy 
# создайте массив 
# t из 51 числа от 0 до 10 включительно.
#%%
t = np.linspace(0,10,51)
t
# Создайте массив Numpy под названием f, содержащий 
# косинусы элементов массива t.
#%%
f = np.array(np.cos(t))
f
# Постройте линейную диаграмму, используя массив 
# t для координат по горизонтали,
# а массив f - для координат по вертикали. 
# Лния графика должна быть зеленого цвета.
#%%
plt.plot(t,f, color = 'green')
plt.title('График f(t)')
plt.xlabel('Значения t')
plt.ylabel('Значения f')
plt.axis([0.5, 9.5, -2.5, 2.5])
# Выведите название диаграммы - 'График f(t)'.
# Также добавьте названия для горизонтальной оси - 
# 'Значения t'
# и для вертикальной - 'Значения f'.
# Ограничьте график по оси x значениями 0.5 и 9.5,
# а по оси y - значениями -2.5 и 2.5.

# *Задание 3
# С помощью функции linspace библиотеки Numpy создайте 
# массив x
# из 51 числа от -3 до 3 включительно.
# Создайте массивы y1, y2, y3, y4 по следующим формулам:
# y1 = x**2
# y2 = 2 * x + 0.5
# y3 = -3 * x - 1.5
# y4 = sin(x)
#%%
x=np.linspace(-3,3,51)
y1 = np.array(x**2)
y2 = np.array(2 * x + 0.5)
y3 = np.array(-3 * x - 1.5)
y4 = np.array(np.sin(x))


# Используя функцию subplots модуля matplotlib.pyplot,
# создайте объект matplotlib.figure.Figure с названием fig
# и массив объектов Axes под названием ax,
# причем так, чтобы у вас было 4 отдельных графика в сетке,
# состоящей из двух строк и двух столбцов.
# В каждом графике массив x используется для координат по горизонтали.
#%%
fig, ax = plt.subplots (nrows = 2, ncols = 2)
ax1, ax2, ax3, ax4 = ax.flatten()
ax1.plot(x, y1 )
ax1.set_title('График y1')
ax1.set_xlim([-5,5])
ax2.plot(x, y2 )
ax2.set_title('График y2')
ax3.plot(x, y3 )
ax3.set_title('График y3')
ax4.plot(x, y4 )
ax4.set_title('График y4')
fig.set_size_inches(8, 6)
plt.subplots_adjust(wspace=0.3, hspace=0.3)

# В левом верхнем графике для координат по вертикали используйте y1,
# в правом верхнем - y2, в левом нижнем - y3, в правом нижнем - y4.

# Дайте название графикам: 'График y1', 'График y2' и т.д.

# Для графика в левом верхнем углу установите границы по оси x от -5 до 5.
# Установите размеры фигуры 8 дюймов по горизонтали и 6 дюймов по вертикали.
# Вертикальные и горизонтальные зазоры между графиками должны составлять 0.3.

# * Задание 4
# Данный датасет является примером несбалансированных данных, 
# так как мошеннические операции с картами встречаются реже обычных.
# Импортруйте библиотеку Pandas, а также используйте для графиков стиль “fivethirtyeight”.
# Посчитайте с помощью метода value_counts количество наблюдений для 
# каждого значения целевой переменной Class и примените к полученным данным 
# метод plot, чтобы построить столбчатую диаграмму. Затем постройте такую же 
# диаграмму, используя логарифмический масштаб.


#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.style as style
import os
%matplotlib inline
%config InlineBackend.figure_format='svg'
try:
	os.chdir(os.path.join(os.getcwd(), 'HW3'))
except:
	pass
#%%
data = pd.read_csv('Data/creditcard.csv')
data.shape
style.use('fivethirtyeight')
#%%
data_info = data['Class'].value_counts()
data_info

#%%
plt.plot(data['Class'], data['Class'].values)
data_info.plot(kind = 'bar')
plt.show()
# data.plot(kind = 'bar')
# plt.show()
# #%%
plt.plot(data['Class'], data['Class'].values)
data_info.plot(kind = 'bar', logy=True)
plt.show()

# На следующем графике постройте две гистограммы по значениям признака V1 - 
# одну для мошеннических транзакций (Class равен 1) и другую - для обычных 
# (Class равен 0). Подберите значение аргумента density так, чтобы по вертикали графика 
# было расположено не число наблюдений, а плотность распределения. 
# Число бинов должно равняться 20 для обеих гистограмм, а коэффициент 
# alpha сделайте равным 0.5, чтобы гистограммы были полупрозрачными и 
# не загораживали друг друга. Создайте легенду с двумя значениями: “Class 0” 
# и “Class 1”. Гистограмма обычных транзакций должна быть серого цвета, 
# а мошеннических - красного. Горизонтальной оси дайте название “Class”.
#%%
data.columns
V1 = data.set_index('V1')['Class']
plt.plot(V1)
plt.show()
