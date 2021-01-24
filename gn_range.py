
import numpy as np
import math as m
import numpy.matlib as matlab

def pos_range(bs_pos, obs_range, lb=np.array([[-m.inf,-m.inf,-m.inf]]), ub=np.array([[m.inf,m.inf,m.inf]]), init_pos=np.array([[0.,0.,0.]]), mask=np.array([0,1,2])):
    
    #check this
    if lb[0,0]!=-m.inf and lb[0,1]!=-m.inf and lb[0,2]!=-m.inf and ub[0,0]!=m.inf and ub[0,1]!=m.inf and ub[0,2]!=m.inf and init_pos[0,0]==0 and init_pos[0,1]==0 and init_pos[0,2]==0 :
        init_pos = (lb+ub)/2
     
    num_bs = bs_pos.shape[0]
    itera = 1
    restart = 0
    done = 0
    est_pos = init_pos
    best_err = m.inf
    tol = 1e-10
    while (not done):
        rel_pos = matlab.repmat(est_pos,num_bs,1)-bs_pos
        
        est_range = np.array([np.sum(rel_pos**2,1)**0.5]).transpose()
        
        err = est_range - obs_range
        
        J = np.divide(rel_pos[:,mask],matlab.repmat(est_range,1,len(mask)))
        
        update = np.matmul(np.matmul(np.linalg.inv(np.matmul(J.transpose(),J)),J.transpose()),err)
        
        tot_err = np.sum(err**2)
        if (tot_err < best_err):
            best_pos = est_pos
            best_err = tot_err
        
        if (abs(update).max() > tol):
        
            est_pos[0,mask] = est_pos[0,mask] - update.transpose()
            #
            
            #
            itera += 1
            
            if (itera >= 30):
                rel_pos = matlab.repmat(est_pos,num_bs,1)-bs_pos
                est_range = np.sum(rel_pos**2)**0.5
                err = est_range - obs_range
                tot_err = np.sum(err**2)
                
                if (tot_err < best_err):
                    best_pos = est_pos
                    best_err = tot_err
                    
                est_pos = np.multiply((ub-lb),np.random.rand(1,3))+lb
                restart += 1
                itera = 1
                
                if (restart >= 40):
                    est_pos = best_pos
                    done = 1
        else:
            done = 1
    
    est_pos = np.multiply(est_pos,(abs(est_pos)>tol))
        
        
    return est_pos 
        
        
