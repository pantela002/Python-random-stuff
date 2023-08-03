import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

X_start=-2
X_end=2

Y_start=-2
Y_end=0

dim_l=X_start+Y_start
dim_r=X_end+Y_end

def u(x):
    return np.heaviside(x, 1)

def xes():
    return np.linspace(dim_l,dim_r, 10000)



def f1(t):
    return (u(t+2)-u(t+1))+t*t*(u(t+1)-u(t))-t*t*(u(t)-u(t-1))-(u(t-1)-u(t-2))

def f2(t):
    return 2*(u(t+2)-u(t+1))-(u(t+1)-u(t))

if __name__ == "__main__":

    x_t= np.array([ f1(t) for t in xes()]) 
    y_t= np.array([ f2(t) for t in xes()])
    y_t=y_t[::-1]
    ys=int((-Y_end-dim_l)/(dim_r-dim_l)*len(x_t))
    y_t=np.roll(y_t,ys)
    print(ys)
    conv=[]



    conv=[]
    print(x_t.size)
    print(y_t.size)


    for i in range(x_t.size):
        conv.append(np.dot(x_t,y_t)*np.count_nonzero(np.dot(x_t,y_t))*6/10000)
        y_t=np.roll(y_t,1)

    



    plt.figure(figsize=(10, 6))
    plt.plot(xes(),conv)
    plt.plot(xes(),y_t)
    plt.plot(xes(),x_t)
    plt.ylim(-2, 2)
    plt.xlim(dim_l, dim_r)
    plt.title('Convolution')
    plt.grid(True)

    plt.show()
