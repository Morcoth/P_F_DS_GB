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
#%%
a=np.arange(20, 0, -2)
print (a)
b=[]
for i in range(20,0, -2):
    b.append([i])    
print(b)

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

#%%
a = np.array([[1,6],[2,8],[3,11],[3,10],[1,7]])
mean_b = np.mean(a[:,0])
mean_b
mean_c=np.mean(a[:,1])
mean_c
mean_a = np.stack((mean_b, mean_c))
mean_a


#%%

a_centred = np.column_stack((a[:,0]-mean_b, a[:,1]-mean_c ))
a_centred
#%%

a_centered_sp = np.dot(a[:,0], a[:,1])
print (a_centered_sp)
#%%
arraysize = a_centered_sp/((np.size(a_centred))-1)
arraysize

#%%
# * Задание 5

cov=np.cov((np.transpose(a)))
cov

