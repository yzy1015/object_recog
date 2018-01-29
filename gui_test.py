#!/usr/bin/env python
import tkinter as tk
from numpy import random
from PIL import ImageTk, Image
import matplotlib.pyplot as plt
from scipy.misc import imsave
from scipy.stats import multivariate_normal
import numpy as np
from tkinter import messagebox  # import this to fix messagebox error
import pickle


window = tk.Tk()
window.title('Color generation')
window.geometry('450x200')

tk.Label(window, text='color: ').place(x=50, y= 50)
tk.Label(window, text='other: ').place(x=50, y= 100)

var_usr_name = tk.StringVar()
var_usr_name.set('1')
entry_usr_name = tk.Entry(window, textvariable=var_usr_name)
entry_usr_name.place(x=160, y=50)
var_usr_pwd = tk.StringVar()
entry_usr_pwd = tk.Entry(window, textvariable=var_usr_pwd)
entry_usr_pwd.place(x=160, y=100)

def usr_generate():

    m1 = np.array([90.595208045488363, 94.424839249654582, 98.290499787437554])
    cov1 = np.array([[ 1435.54436528,  1448.82313544,  1449.18143498],
       [ 1448.82313544,  1469.37034307,  1468.21597487],
       [ 1449.18143498,  1468.21597487,  1470.86624136]])

    m2 = np.array([118.41996558993118, 147.10866296732593, 123.63691177382354])
    cov2 = np.array([[ 1463.7264814 ,  1455.4160082 ,  1458.66421291],
       [ 1455.4160082 ,  1452.36768879,  1453.54715988],
       [ 1458.66421291,  1453.54715988,  1455.55414233]])

    m3 = np.array([137.52923748635584, 84.341131957786217, 78.770778106970212])
    cov3 = np.array([[ 723.05062757,  677.25175499,  655.1176708 ],
       [ 677.25175499,  660.38979244,  643.14910085],
       [ 655.1176708 ,  643.14910085,  640.02131679]])
    

    m_l = [m1,m2,m3]
    cov_l = [cov1,cov2,cov3]
    usr_name = var_usr_name.get()
    print(usr_name)
    
    mn = multivariate_normal(mean=m_l[int(usr_name)], cov=cov_l[int(usr_name)])
    random_g = mn.rvs(size=9000, random_state=12345)
    x0 = np.reshape(random_g.T[0],(300,300)) 
    x1 = np.reshape(random_g.T[1],(300,300))
    x2 = np.reshape(random_g.T[2],(300,300))
    pic = np.array([x0,x1,x2]).T.astype(np.uint8)
    plt.imshow(pic)
    plt.show()


def usr_sign_clear():
    pass

# login and sign up button
btn_login = tk.Button(window, text='Generate', command=usr_generate)
btn_login.place(x=150, y=150)
btn_sign_up = tk.Button(window, text='Clear', command=usr_sign_clear)
btn_sign_up.place(x=250, y=150)

window.mainloop()
