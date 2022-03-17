import numpy as np
from numpy.random import random
from scipy import interpolate
import matplotlib.pyplot as plt

# some base function
def f(x):
    return np.exp(-(0.5*x)**2)

def inverse_cdf(base_function,x_begin,x_end,num):
    x = np.linspace(x_begin,x_end,int(num))
    # probability density function, PDF
    y = base_function(x)
    # cumulative distribution function, CDF                          
    cdf_y = np.cumsum(y)
    # normalizing CDF  
    cdf_y = cdf_y/cdf_y.max()
    # return the inverse "function" for normal CDF     
    return interpolate.interp1d(cdf_y,x)           

# generate samples according to the chosen, PDF and base function
def return_samples(base_function,x_begin,x_end,num):
    uniform_samples = random(int(num))
    required_samples = inverse_cdf(base_function,x_begin,x_end,num)
    return required_samples(uniform_samples)



if __name__ == '__main__':
    y = return_samples(f,-5,5,1e6)
    x = np.linspace(-5,5,int(1e6))

    # plot
    fig,ax = plt.subplots()
    ax.set_xlabel('x')
    ax.set_ylabel('Probability Density')
    ax.plot(x,f(x)/np.sum( f(x)*(x[1]-x[0]) ))
    ax.hist(y,bins='auto',density=True,range=(x.min(),x.max()))
    plt.show() 