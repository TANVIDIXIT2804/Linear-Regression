# -*- coding: utf-8 -*-
"""Q5_test.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/19fLBYgrxvEcH2qfm0URlFaLuEUyLMibB
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from preprocessing.polynomial_features import PolynomialFeatures
from linearRegression.linear_regression import LinearRegression
import os.path
from os import path

if not path.exists('Plots/Question5/'):
    os.makedirs('Plots/Question5/')

N_values = [50,70,90, 130]

for N in N_values:
  x = np.array([i*np.pi/180 for i in range(N,N*5,4)])
  np.random.seed(10)
  y = 3*x + 8 + np.random.normal(0,3,len(x))

  Y = pd.Series(y)
  X = x.reshape(-1,1)
  LR = LinearRegression(fit_intercept=False)
  coeffs = []
  degrees = []
  list=[1,3,5,7,9]
  for deg in list:
    poly = PolynomialFeatures(degree=deg, include_bias=True)
    x_new=np.array([poly.transform(X[0])])
    for i in range(1,len(X)):
      x_new=np.concatenate((x_new,np.array([poly.transform(X[i])])))
    x_new=pd.DataFrame(x_new)
    LR.fit_SVD(x_new, y)
    coeffs.append((np.linalg.norm(LR.coef_)))
    degrees.append(deg)
  plt.plot(degrees,coeffs,label=N)
plt.xlabel('Degree')
plt.ylabel('Max Absolute Value of thetha_i')
plt.title('Norm of Theta v/s Dataset Size')
plt.legend()
plt.savefig('Plots/Question5/norm_vs_size.png')
plt.show()


# TODO : Write here
