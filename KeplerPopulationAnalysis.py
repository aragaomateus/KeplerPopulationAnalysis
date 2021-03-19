#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 22:29:19 2021

@author: aragaom
"""

#%%
# 1) Caste membership (1 to 5)
# 2) Intelligence (in alien IQ points)
# 3) Brain mass (in g)
# 4) Hours worked per week
# 5) Annual income (in credits)

# loading data and libraries:
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn import linear_model 
keplerData = np.genfromtxt('/Users/aragaom/Desktop/NYU Classes/Intro to Data Science Class/Lab5/kepler.txt')
def simple_linear_regress_func(data):
    # Load numpy:
    import numpy as np
    # Initialize container:
    temp_cont = np.empty([len(data),5]) # initialize empty container, n x 5
    temp_cont[:] = np.NaN # convert to NaN
    # Store data in container:
    for i in range(len(data)):
        temp_cont[i,0] = data[i,0] # x 
        temp_cont[i,1] = data[i,1] # y 
        temp_cont[i,2] = data[i,0]*data[i,1] # x*y
        temp_cont[i,3] = data[i,0]**2 #x^2
        temp_cont[i,4] = data[i,1]**2 # y^2
    # Compute numerator and denominator for m:
    m_numer = len(data)*sum(temp_cont[:,2]) - sum(temp_cont[:,0])*sum(temp_cont[:,1])
    m_denom = len(data)*sum(temp_cont[:,3]) - (sum(temp_cont[:,0]))**2
    m = m_numer/m_denom
    # Compute numerator and denominator for b:
    b_numer = sum(temp_cont[:,1]) - m * sum(temp_cont[:,0])
    b_denom = len(data)
    b = b_numer/b_denom
    # Compute r^2:
    temp = np.corrcoef(data[:,0],data[:,1]) # pearson r
    r_sqr = temp[0,1]**2 # L-7 weenie it (square)
    # Output m, b & r^2:
    output = np.array([m, b, r_sqr])
    return output 
#%%
# There are concerns that the IQ test is not culturally fair, 
# privileging members of higher castes. You look for evidence 
# of this and determine that the correlation between caste membership and IQ is
data1 = [keplerData[:,0],keplerData[:,1]]
correlationOfCaste_IQ = np.corrcoef(data)

#%%
# However, you realize that this correlation is not in itself – without first considering 
# other possible confounds – genuine evidence for oppression. So you correlate caste membership 
# and IQ while controlling for brain mass and determine that it is closest to
data2 = np.array([keplerData[:,0],keplerData[:,1],keplerData[:,2]])
inputData = np.transpose([data2[2,:],data2[0,:]]) #brain mass and caste
output = simple_linear_regress_func(inputData) # output returns m,b,r^2
y = data2[0,:] # caste membership
yHat = output[0]*data2[2,:] + output[1] # predicted income
residuals1 = y - yHat # compute residuals

inputData = np.transpose([data2[2,:],data2[1,:]])#brain mass and IQ
output = simple_linear_regress_func(inputData) # output returns m,b,r^2
y = data2[1,:] # IQ
yHat = output[0]*data2[2,:] + output[1] # predicted violent incidents
residuals2 = y - yHat # compute residuals

# 4. Correlate the residuals:
part_corr = np.corrcoef(residuals1,residuals2) # r = -0.09

#%%
# You now realize that brain mass is a good predictor of intelligence. 
# What proportion of the variance in intelligence does brain mass account for?
data3 = np.array([keplerData[:,1],keplerData[:,2]])
inputData = np.transpose([data2[2,:],data2[1,:]])
output = simple_linear_regress_func(inputData) #this correspondes to the R^2
#%%
# When predicting intelligence from brain mass, what intelligence score would 
# you expect given a brain mass of 3000 g?
prediction = np.array([3000]).reshape(1,-1)
data3 = np.array([keplerData[:,1],keplerData[:,2]])
inputData = np.transpose([data3[0,:],data3[1,:]])
#regr = linear_model.LinearRegression() # linearRegression function from linear_model
regr.fit(data3[1,:],data3[0,:])
X = np.array(data3[1,:]).reshape(-1,1)
regr.predict(X)
regr.predict([[3000]])


#%%
# Concerns have arisen that suggest that the society on this planet discriminates 
# against members of the lower castes when it comes to income. You look for evidence 
# for this and find that the correlation between income and caste membership is



#%%
# However, contrarians argue that this correlation is spurious and that it is not due 
# to caste membership per se. To investigate this claim, you correlate caste membership 
# and income while controlling for intelligence and hours worked and find that it is closest to

#%%
# What proportion of the variance in income does intelligence account for?

#%%
# What proportion of the variance in income do hours worked account for?

#%%
# What proportion of the variance in income if accounted for by intelligence and "hours worked" together?

#%%
# What income (in credits) would we predict for someone with an alien IQ of 120 who is working 50 hours a week?
