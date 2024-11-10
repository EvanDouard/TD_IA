
import numpy as np
import matplotlib.pyplot as plt
max_epsilon=1
min_epsilon=0.01
epsilon_step=0.0009

def function(e):
    return min_epsilon+ ((max_epsilon-min_epsilon)*np.exp(-epsilon_step*e))

L=[]
L2=[]
for i in range(0,10000):
    L.append(i)
    L2.append(function(i))


print(L2)
plt.plot(L,L2)
plt.show()