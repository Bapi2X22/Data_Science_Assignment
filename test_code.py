import numpy as np
import matplotlib.pyplot as plt
x = np.linspace (-12,12,1000)
def fnc(x,Np):   # np is number of data points to be generated from the target function. 
#If you want to generate a single data point put np=1. If you want to plot the target function put higher value of np.
    nos = np.random.normal(0,0.2,Np)
    return (x-2)**2 + 4 + nos
plt.title("The target function")
# plt.xlim(0,4)
# plt.ylim(0,10)
plt.xlabel("x")
plt.ylabel("y")
# plt.scatter(x,y)
plt.plot(x,fnc(x,1000))
plt.show()
