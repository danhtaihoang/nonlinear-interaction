"""
functions for generating binary variables
"""
import numpy as np
import function as ft

#=========================================================================================
"""generate quandratic term with standart deviation g/n then make it to be 
   symmetry: q[i,j,k] = q[i,k,j] 
   and no-self interaction: q[i,j,j] = 0  
""" 
def generate_interaction(g,n):
    q = np.random.normal(0.0,g/n,size=(n,n,n))
    
    # make q to be symmetry and no-self interaction
    for j in range(n):
        for k in range(n):
            if k > j: q[:,j,k] = q[:,k,j]
            if k == j: q[:,j,j] = 0.
    return q 
#=========================================================================================
"""generate binary time-series 
    input: interaction matrix w[n,n], interaction variance g, data length l
    output: time series s[l,n]
""" 
def generate_data(w,q,l):
    n = np.shape(w)[0]

    s = np.ones((l,n))
    for t in range(1,l-1):    
        # calculate h1[t,i] = sum_j w[i,j]*s[t,j]
        h1 = np.sum(w[:,:]*s[t,:],axis=1) # Wij from j to i

        # calculate h2[t,i] = sum q[i,j,k]*s[t,j]*s[t,k]
        h21 = np.sum(q[:,:,:]*s[t,:],axis=1)
        h2 = np.sum(h21*s[t,:],axis=1)

        h = h1 + 0.5*h2

        p = 1/(1+np.exp(-2*h))
        s[t+1,:]= ft.sign_vec(p-np.random.rand(n))

    return s


