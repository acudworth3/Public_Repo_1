import numpy as np
import time 

#Small Data Test Case
X_small = np.arange(1,4)*np.ones((2,3))
ex_sig = np.arange(1,10).reshape(3,3) #not that ex_sigma is equivalent to sigma.inverse() for our purposes

single_row_solution = np.dot(ex_sig,X_small[0]) #[14,32,50]


X_tensor = X_small[:,:,np.newaxis] #add a new dimension
pointwise_mult =  X_tensor*ex_sig.transpose() #I'm not sure why transpose is needed here. Something to do with axes
matmul_vec = np.sum(pointwise_mult,axis=1) #sum over the pointwise multiplication

#check that everything worked
assert(np.equal(matmul_vec[:,:],single_row_solution).all())
print("small vector success!")


#LARGE DATA Test Case
start_time = time.time()
X_Large = np.arange(1,4)*np.ones((1000000,3))
ex_sig = np.arange(1,10).reshape(3,3) #not that ex_sigma is equivalent to sigma.inverse() for our purposes

single_row_solution = np.dot(ex_sig,X_Large[0]) #[14,32,50]

X_tensor = X_Large[:,:,np.newaxis] #add a new dimension
pointwise_mult =  X_tensor*ex_sig.transpose() #I'm not sure why transpose is needed here. Something to do with axes
matmul_vec = np.sum(pointwise_mult,axis=1) #sum over the pointwise multiplication

#check that everything worked
assert(np.equal(matmul_vec[:,:],single_row_solution).all())
print("Large vector success!")
end_time = time.time()
print("Large Vector Run Time (s): ",round(end_time-start_time,5 ))


start_time = time.time()
matmul_vec = np.ones((X_Large.shape[0],X_Large.shape[1]))
for row in range(X_Large.shape[0]):
    matmul_vec[row,:] *= np.dot(ex_sig,X_Large[row,:])
assert(np.equal(matmul_vec[:,:],single_row_solution).all())    
end_time = time.time()
print("For Loop Run Time (s): ",round(end_time-start_time,5 ))


#RESULTS:
#vectorized runtime ~0.378 s
#loop run time ~7.725 s