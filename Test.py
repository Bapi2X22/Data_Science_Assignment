#### Problem 2.23
import numpy as np
from functools import partial
import matplotlib.pyplot as plt
import math
from scipy.optimize import minimize
# np.random.seed(123456)

from matplotlib.font_manager import FontProperties

def sample(lb, ub, sz):
    # sample randomly from a uniform distribution 
    return lb + np.random.random_sample((sz,))*(ub-lb)

def h_func0(x,x_1,x_2,x_3):
    b = ((((x_1-2)**2)+((x_2-2)**2)+((x_3-2)**2)+12))/3
    return b

def h_func1(x,x_1,x_2,x_3):
    term_1 = (((x_1-2)**2)+((x_2-2)**2)+((x_3-2)**2)+12)/3
    term_2 = ((x_1*(x_1-2)**2)+(x_2*(x_2-2)**2)+(x_3*(x_3-2)**2)+4*(x_1+x_2+x_3))/(x_1+x_2+x_3)
    term_3 = (((x_1+x_2+x_3)/3)-(((x_1**2)+(x_2**2)+(x_3**2))/(x_1+x_2+x_3)))
    a = (term_1-term_2)/term_3
    b = ((((x_1-2)**2)+((x_2-2)**2)+((x_3-2)**2)+12) -(a*(x_1+x_2+x_3)))/3
    return a*x+b

def avg_g(x, gdfunc, num_samples, targetfunc):
    #compute the average hypothesis \bar{g} at given point x
    bias_at_x = 0 
    gd_funcs = []
    for i in range(num_samples):
        #generate 2 sample data each time
        x1, x2, x3 = sample(-10, 10, 3)
        v = gdfunc(x, x1, x2, x3)
        gd_funcs.append(v)
        
    average_gfunc_at_x = np.mean(gd_funcs)
    #print('x: ', x, 'average_gfunc_at_x: ', average_gfunc_at_x)
    variance_gfunc_at_x = np.var(gd_funcs)
    bias_at_x = (average_gfunc_at_x - targetfunc(x))**2
    return average_gfunc_at_x, variance_gfunc_at_x, bias_at_x

# Compute the expected value of variance, bias and out-of-sample error
def calc_bias_var_eout(gd_func, target_func, num_data_samples, num_x_samples):
    variances, biases, eouts = [], [], []
    for i in range(num_x_samples):
        x = sample(-10, 10, 1)
        _, variance, bias = avg_g(x, gd_func, num_data_samples, target_func)
        variances.append(variance)
        biases.append(bias)

        # Compute the expected value of out-of-sample error w.r.t. data
        eout_on_data = []
        for i in range(num_data_samples):
            x1, x2, x3 = sample(-10, 10, 3)
            v= gd_func(x, x1, x2, x3)
            eout_on_data.append((v-target_func(x))**2) # (g^{D}(x) - f(x))**2

        eout_data_avg = np.mean(eout_on_data)
        eouts.append(eout_data_avg)



    variance = np.mean(variances)    
    bias = np.mean(biases)
    eout = np.mean(eouts)
    print('The variance is: ', variance)
    print('The bias is: ', bias)
    print('The expected out-of-sample error is: ', eout)
    print('The variance+bias is: ', variance+bias)


    xs = np.arange(-10, 10, 0.01)
    true_f, avg_gf, var_gf, ubs, lbs = [],[], [], [], []
    for x in xs:
        true_f.append(target_func(x))
        mean_g, var_g, bias_g = avg_g(x, gd_func, num_data_samples, target_func)
        avg_gf.append(mean_g)
        var_gf.append(var_g)
        ubs.append(mean_g + np.sqrt(var_g))
        lbs.append(mean_g - np.sqrt(var_g))
        
    plt.plot(xs, true_f, color='red', label='Problem 2.23: True Function')
    plt.plot(xs, avg_gf, color='green', label='Problem 2.23: Average Hypothesis g_bar')
    plt.plot(xs, ubs, color='blue', label='Problem 2.23: Upper bound of the average hypothesis')
    plt.plot(xs, lbs, color='blue', label='Problem 2.23: Lower bound of the average hypothesis')
    legend_x = 2.0
    legend_y = 0.5
    plt.legend(['Problem 2.23: True Function', 
                'Problem 2.23: Average Hypothesis g_bar',
                'Problem 2.23: Upper bound of the average hypothesis',
                'Problem 2.23: Lower bound of the average hypothesis'], 
               loc='center right', bbox_to_anchor=(legend_x, legend_y))
    
num_data_samples = 1000
num_x_samples = 1000
print('------ Hypothesis set: h(x) = ax + b ------')
# def fnc(x,Np):   # Np is number of data points to be generated from the target function. 
# #If you want to generate a single data point put Np=1. If you want to plot the target function put higher value of Np.
#     Noise = np.random.normal(0,0.2,Np)
#     return (x-2)**2 + 4 + Noise
# Noise = np.random.normal(0,0.2,1)
calc_bias_var_eout(h_func0, lambda x: (x-2)**2 + 4 + np.random.normal(0,0.2,1), num_data_samples, num_x_samples)    
# calc_bias_var_eout(h_func0, lambda x: (x-2)**2 + 4 + Noise, num_data_samples, num_x_samples) 
plt.show()
