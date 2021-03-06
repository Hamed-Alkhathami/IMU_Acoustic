
import numpy as np

def q_conj_mat(q):
    qConj = np.hstack((q[:,0:1], -q[:,1:2], -q[:,2:3], -q[:,3:4]))
    return qConj
    
def q_prod_mat(a,b):
    ab = np.zeros((len(a),4))
    ab[:,0] = a[:,0] * b[:,0] - a[:,1] * b[:,1] - a[:,2] * b[:,2] - a[:,3] * b[:,3] 
    ab[:,1] = a[:,0] * b[:,1] + a[:,1] * b[:,0] + a[:,2] * b[:,3] - a[:,3] * b[:,2] 
    ab[:,2] = a[:,0] * b[:,2] - a[:,1] * b[:,3] + a[:,2] * b[:,0] + a[:,3] * b[:,1] 
    ab[:,3] = a[:,0] * b[:,3] + a[:,1] * b[:,2] - a[:,2] * b[:,1] + a[:,3] * b[:,0] 
    return np.hstack((ab[:,0:1],ab[:,1:2],ab[:,2:3],ab[:,3:4]))
    
def q_product(a,b):
    ab = np.zeros(4)
    ab[0] = a[0,0] * b[0,0] - a[0,1] * b[0,1] - a[0,2] * b[0,2] - a[0,3] * b[0,3] 
    ab[1] = a[0,0] * b[0,1] + a[0,1] * b[0,0] + a[0,2] * b[0,3] - a[0,3] * b[0,2] 
    ab[2] = a[0,0] * b[0,2] - a[0,1] * b[0,3] + a[0,2] * b[0,0] + a[0,3] * b[0,1] 
    ab[3] = a[0,0] * b[0,3] + a[0,1] * b[0,2] - a[0,2] * b[0,1] + a[0,3] * b[0,0] 
    return np.hstack((ab[0],ab[1],ab[2],ab[3]))
