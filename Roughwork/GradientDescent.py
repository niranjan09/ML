import numpy as np

def derivative_f(x):
    return 2*x

def f(x):
    return x*x

def get_minima(x, f, derivative_f, alpha = 0.01, epoch = 1000):
    for i in range(epoch):
        x = x - alpha * derivative_f(x)
    return x

print get_minima(4, f, derivative_f, 0.01, 1000)
