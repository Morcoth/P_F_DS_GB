#%%
import numpy as np

a = np.arange(12,24,1,dtype = int)
a

#%%
a1= a.reshape((2, 6)).copy()
a1

#%%
a2= a.reshape((3, 4))
a2

#%%
a3= a.reshape((4, 3))
a3
#%%
a4= a.reshape((6, 2))
a4
#%%
a5= np.dot(a4, a1)
a5



#%%
a6 = a.reshape(2,-1)
a6
#%%
a7 = a.reshape (3,-1)
a7
#%%
a8=a.reshape(-1, 2)
a8
#%%
a9=a.reshape(-1, 3)
a9
#%%
a10=a.reshape(-1, 4)
a10
#%%
print('Да, вектор это одномерный массив')
#%%
a=np.arange(20, 0, -2)
print (a)
b=[]
for i in range(20,0, -2):
    b.append([i])    
print(b)
print('Если я правильно понял задание(мне кажется что нет), то разница в том что первый массив единичен, второй является массивом массивов столбцов')


#%%
a=np.zeros((5,5), dtype=int)
b=np.ones((4,5))
np.vstack((a,b))

#%%
a=np.arange(0,12)
A=a.reshape(4,3)
At=A.transpose()
B=np.dot(A,At)
print(B)
np.shape(B)
#4/4
np.linalg.det(B)
#определитель равен нулю следовательно обратная матрица невозможна
#%%
np.random.seed(42)
A=np.random.randint(0,16,16)
C=A.reshape(4,4)
D=np.add(B,C*10)
#%%
np.linalg.det(D)
#%%
np.linalg.matrix_rank(D)
#%%
D_inv=np.linalg.inv(D)
D_inv
#%%
#10 
for index, i in np.ndenumerate(D_inv):       
    if i<0:
        D_inv[index]=0
    elif i>0:
        D_inv[index]=1
D_inv


# Задание 2
# Создайте массив Numpy под названием a размером 5x2, 
# то есть состоящий из 5 строк и 2 столбцов.
# Первый столбец должен содержать числа 1, 2, 3, 3, 1, 
# а второй - числа 6, 8, 11, 10, 7.
# Будем считать, что каждый столбец - это признак, 
# а строка - наблюдение.
# Затем найдите среднее значение по каждому признаку, 
# используя метод mean массива Numpy.
# Результат запишите в массив mean_a, в нем должно быть 2 элемента.
#%%
a = np.array([[1,6],[2,8],[3,11],[3,10],[1,7]])
mean_b = np.mean(a[:,0])
mean_b
mean_c=np.mean(a[:,1])
mean_c
mean_a = np.stack((mean_b, mean_c))
mean_a

#%%
#Задание 3
# Вычислите массив a_centered, отняв от значений массива а
# средние значения соответствующих признаков, содержащиеся в массиве mean_a.
# Вычисление должно производиться в одно действие.
# Получившийся массив должен иметь размер 5x2.
#%%

a_centred = np.column_stack((a[:,0]-mean_b, a[:,1]-mean_c ))
a_centred
#%%
#Задание 4
# Найдите скалярное произведение столбцов массива a_centered.
# В результате должна получиться величина a_centered_sp.
# Затем поделите a_centered_sp на N-1, где N - число наблюдений.

a_centered_sp = np.dot(a[:,0], a[:,1])
print (a_centered_sp)
#%%
arraysize = a_centered_sp/((np.size(a_centred))-1)
arraysize