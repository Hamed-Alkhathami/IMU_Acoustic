import numpy as np
import math as m
import matplotlib.pyplot as plt
from qfunc import q_conj_mat, q_prod_mat, q_product
    
f = 256
delta_t = 1/f

w1 = 5
w1 = w1*m.pi/180

h = 1
r = 5

p1b = np.array([[2,3,0]])
p2b = np.array([[-4,5,0]])
p3b = np.array([[-3,-6,0]])
p4b = np.array([[6,-2,0]])

t_end = 2*m.pi/w1
t = np.arange(0,t_end*f,dtype=np.float)*delta_t
theta = w1*t

x = r*np.cos(theta)
y = r*np.sin(theta)
z = h*np.ones(len(t))

p = np.array([x,y,z])
p = p.transpose()

meu = theta
e = np.array([[0,0,1]])

qsca = np.array([np.cos(meu/2)]).transpose()
qvec = np.matmul(np.array([np.sin(meu/2)]).transpose(),e)

q = np.hstack((qsca,qvec))

q_norm = np.zeros(q.shape)
i = 0
for x in q_norm[:,1]: 
    q_norm[i,:] = q[i,:]/np.linalg.norm(q[i,:])   
    i += 1

dp = np.array([np.diff(p[:,0]),np.diff(p[:,1]),np.diff(p[:,2])]).transpose()
dt = np.array([np.diff(t)]).transpose()
dq = np.array([np.diff(q[:,0]),np.diff(q[:,1]),np.diff(q[:,2]),np.diff(q[:,3])]).transpose()
v = np.hstack((np.divide(dp[:,0:1],dt), np.divide(dp[:,1:2],dt), np.divide(dp[:,2:3],dt)))
dv = np.array([np.diff(v[:,0]),np.diff(v[:,1]),np.diff(v[:,2])]).transpose()
a = np.hstack((np.divide(dv[:,0:1],dt[0:-1]), np.divide(dv[:,1:2],dt[0:-1]), np.divide(dv[:,2:3],dt[0:-1])))

w = 2*q_prod_mat(np.divide(dq,dt),q_conj_mat(q_norm[0:-1,:]))
w = w[:,1:4]

dw = np.array([np.diff(w[:,0]),np.diff(w[:,1]),np.diff(w[:,2])]).transpose()
alpha = np.hstack((np.divide(dw[:,0:1],dt[0:-1]), np.divide(dw[:,1:2],dt[0:-1]), np.divide(dw[:,2:3],dt[0:-1])))

p1 = q_prod_mat(q_prod_mat(q_conj_mat(q_norm),np.hstack((np.array([[0]]),p1b))),q)   
p1 = p1[:,1:4]
p2 = q_prod_mat(q_prod_mat(q_conj_mat(q_norm),np.hstack((np.array([[0]]),p2b))),q)   
p2 = p2[:,1:4]
p3 = q_prod_mat(q_prod_mat(q_conj_mat(q_norm),np.hstack((np.array([[0]]),p3b))),q)     
p3 = p3[:,1:4]
p4 = q_prod_mat(q_prod_mat(q_conj_mat(q_norm),np.hstack((np.array([[0]]),p4b))),q)    
p4 = p4[:,1:4]

r1p = p1 - p
r2p = p2 - p
r3p = p3 - p
r4p = p4 - p

a1 = np.zeros([len(t)-2,3])
a2 = np.zeros([len(t)-2,3])
a3 = np.zeros([len(t)-2,3])
a4 = np.zeros([len(t)-2,3])
i= 0


for x in a1:
    a1[i,:] = a[i,:] + np.cross(alpha[i,:],r1p[i,:]) + np.cross(w[i,:],np.cross(w[i,:],r1p[i,:]))
    a2[i,:] = a[i,:] + np.cross(alpha[i,:],r2p[i,:]) + np.cross(w[i,:],np.cross(w[i,:],r2p[i,:]))
    a3[i,:] = a[i,:] + np.cross(alpha[i,:],r3p[i,:]) + np.cross(w[i,:],np.cross(w[i,:],r3p[i,:]))
    a4[i,:] = a[i,:] + np.cross(alpha[i,:],r4p[i,:]) + np.cross(w[i,:],np.cross(w[i,:],r4p[i,:]))
    i += 1 

#






#

SNRg = 50
SNRa = 50

Ka1 = 5
Ka2 = 7
Ka3 = 3
Ka4 = 1 

I = np.eye(3);

Sax1 = 10
Sax2 = 1
Sax3 = 9
Sax4 = 5
Say1 = 12
Say2 = 15
Say3 = 7
Say4 = 3
Saz1 = 7
Saz2 = 9
Saz3 = 4
Saz4 = 8
Sa1 = np.array([[Sax1,0,0],[0,Say1,0],[0,0,Saz1]])
Sa2 = np.array([[Sax2,0,0],[0,Say2,0],[0,0,Saz2]])
Sa3 = np.array([[Sax3,0,0],[0,Say3,0],[0,0,Saz3]])
Sa4 = np.array([[Sax4,0,0],[0,Say4,0],[0,0,Saz4]])

Maxy1 = 5
Maxy2 = 7
Maxy3 = 12
Maxy4 = 2
Maxz1 = 11
Maxz2 = 3
Maxz3 = 5
Maxz4 = 6
Mayz1 = 4
Mayz2 = 6
Mayz3 = 8
Mayz4 = 3
Ma1 = np.array([[0,Maxy1,Maxz1],[-Maxy1,0,Mayz1],[-Maxz1,-Mayz1,0]])
Ma2 = np.array([[0,Maxy2,Maxz2],[-Maxy2,0,Mayz2],[-Maxz2,-Mayz2,0]])
Ma3 = np.array([[0,Maxy3,Maxz3],[-Maxy3,0,Mayz3],[-Maxz3,-Mayz3,0]])
Ma4 = np.array([[0,Maxy4,Maxz4],[-Maxy4,0,Mayz4],[-Maxz4,-Mayz4,0]])

bax1 = 4
bax2 = 3
bax3 = 12
bax4 = 11
bay1 = 8
bay2 = 5
bay3 = 2
bay4 = 3
baz1 = 16
baz2 = 7
baz3 = 9
baz4 = 1
ba1 = np.array([[bax1,bay1,baz1]])
ba2 = np.array([[bax2,bay2,baz2]])
ba3 = np.array([[bax3,bay3,baz3]])
ba4 = np.array([[bax4,bay4,baz4]])

Tax1 = 3
Tax2 = 15
Tax3 = 5
Tax4 = 11
Tay1 = 6
Tay2 = 3
Tay3 = 7
Tay4 = 9
Taz1 = 4
Taz2 = 8
Taz3 = 2
Taz4 = 1
Ta1 = np.array([[Tax1,Tay1,Taz1]])
Ta2 = np.array([[Tax2,Tay2,Taz2]])
Ta3 = np.array([[Tax3,Tay3,Taz3]])
Ta4 = np.array([[Tax4,Tay4,Taz4]])
dTempA1 = 5
dTempA2 = 15
dTempA3 = 20
dTempA4 = 4

ab1 = np.zeros([len(t)-2,3])
ab2 = np.zeros([len(t)-2,3])
ab3 = np.zeros([len(t)-2,3])
ab4 = np.zeros([len(t)-2,3])
i = 0
for x in ab1:
    ab1[i,:] = (np.matmul((I+Sa1+Ma1),a1[i,:].transpose()).transpose() +ba1+Ta1*dTempA1)/Ka1
    ab2[i,:] = (np.matmul((I+Sa2+Ma2),a2[i,:].transpose()).transpose() +ba2+Ta2*dTempA2)/Ka2
    ab3[i,:] = (np.matmul((I+Sa3+Ma3),a3[i,:].transpose()).transpose() +ba3+Ta3*dTempA3)/Ka3
    ab4[i,:] = (np.matmul((I+Sa4+Ma4),a4[i,:].transpose()).transpose() +ba4+Ta4*dTempA4)/Ka4
    i += 1

#for AWGN see: https://stackoverflow.com/questions/14058340/adding-noise-to-a-signal-in-python
ab = (ab1+ab2+ab3+ab4)/4    
ab_avg_db = 10 * np.log10(np.mean(ab))
na_avg_watts = 10 ** ((ab_avg_db - SNRa) / 10)  
na = np.random.normal(0, np.sqrt(na_avg_watts), [len(ab),3])

abn1 = ab1 + na            
abn2 = ab2 + na         
abn3 = ab3 + na         
abn4 = ab4 + na            


acal1 = np.zeros([len(t)-2,3])
acal2 = np.zeros([len(t)-2,3])
acal3 = np.zeros([len(t)-2,3])
acal4 = np.zeros([len(t)-2,3])
i = 0
for x in abn1: 
    acal1[i,:] = (np.matmul(np.linalg.inv(I+Sa1+Ma1),(Ka1*abn1[i,:]-ba1-Ta1*dTempA1).transpose())).transpose()
    acal2[i,:] = (np.matmul(np.linalg.inv(I+Sa2+Ma2),(Ka2*abn2[i,:]-ba2-Ta2*dTempA2).transpose())).transpose()
    acal3[i,:] = (np.matmul(np.linalg.inv(I+Sa3+Ma3),(Ka3*abn3[i,:]-ba3-Ta3*dTempA3).transpose())).transpose()
    acal4[i,:] = (np.matmul(np.linalg.inv(I+Sa4+Ma4),(Ka4*abn4[i,:]-ba4-Ta4*dTempA4).transpose())).transpose()
    i += 1 

Kg1 = 5
Kg2 = 3
Kg3 = 7
Kg4 = 2 

I = np.eye(3)

Sgx1 = 10
Sgx2 = 8
Sgx3 = 5
Sgx4 = 10
Sgy1 = 9
Sgy2 = 12
Sgy3 = 7
Sgy4 = 13
Sgz1 = 7
Sgz2 = 2
Sgz3 = 3
Sgz4 = 9
Sg1 = np.array([[Sgx1,0,0],[0,Sgy1,0],[0,0,Sgz1]])
Sg2 = np.array([[Sgx2,0,0],[0,Sgy2,0],[0,0,Sgz2]])
Sg3 = np.array([[Sgx3,0,0],[0,Sgy3,0],[0,0,Sgz3]])
Sg4 = np.array([[Sgx4,0,0],[0,Sgy4,0],[0,0,Sgz4]])

Mgxy1 = 5
Mgxy2 = 3
Mgxy3 = 10
Mgxy4 = 2
Mgxz1 = 7
Mgxz2 = 11
Mgxz3 = 11
Mgxz4 = 7
Mgyz1 = 5
Mgyz2 = 9
Mgyz3 = 13
Mgyz4 = 11
Mg1 = np.array([[0,Mgxy1,Mgxz1],[-Mgxy1,0,Mgyz1],[-Mgxz1,-Mgyz1,0]])
Mg2 = np.array([[0,Mgxy2,Mgxz2],[-Mgxy2,0,Mgyz2],[-Mgxz2,-Mgyz2,0]])
Mg3 = np.array([[0,Mgxy3,Mgxz3],[-Mgxy3,0,Mgyz3],[-Mgxz3,-Mgyz3,0]])
Mg4 = np.array([[0,Mgxy4,Mgxz4],[-Mgxy4,0,Mgyz4],[-Mgxz4,-Mgyz4,0]])

bgx1 = 14
bgx2 = 4
bgx3 = 8
bgx4 = 19
bgy1 = 7
bgy2 = 2
bgy3 = 2
bgy4 = 9
bgz1 = 16
bgz2 = 11
bgz3 = 15
bgz4 = 11
bg1 = np.array([[bgx1,bgy1,bgz1]])
bg2 = np.array([[bgx2,bgy2,bgz2]])
bg3 = np.array([[bgx3,bgy3,bgz3]])
bg4 = np.array([[bgx4,bgy4,bgz4]])

Ggxx1 = 3
Ggxx2 = 5
Ggxx3 = 5
Ggxx4 = 3
Ggxy1 = 9
Ggxy2 = 7
Ggxy3 = 8
Ggxy4 = 2 
Ggxz1 = 5
Ggxz2 = 6
Ggxz3 = 1
Ggxz4 = 9
Ggyx1 = 2
Ggyx2 = 7
Ggyx3 = 4
Ggyx4 = 1
Ggyy1 = 2
Ggyy2 = 8
Ggyy3 = 9
Ggyy4 = 5
Ggyz1 = 7
Ggyz2 = 7
Ggyz3 = 2
Ggyz4 = 1
Ggzx1 = 10
Ggzx2 = 11
Ggzx3 = 5
Ggzx4 = 6
Ggzy1 = 7
Ggzy2 = 5
Ggzy3 = 2
Ggzy4 = 6
Ggzz1 = 2
Ggzz2 = 5
Ggzz3 = 1
Ggzz4 = 1
Gg1 = np.array([[Ggxx1,Ggxy1,Ggxz1],[Ggyx1,Ggyy1,Ggyz1],[Ggzx1,Ggzy1,Ggzz1]])
Gg2 = np.array([[Ggxx2,Ggxy2,Ggxz2],[Ggyx2,Ggyy2,Ggyz2],[Ggzx2,Ggzy2,Ggzz2]])
Gg3 = np.array([[Ggxx3,Ggxy3,Ggxz3],[Ggyx3,Ggyy3,Ggyz3],[Ggzx3,Ggzy3,Ggzz3]])
Gg4 = np.array([[Ggxx4,Ggxy4,Ggxz4],[Ggyx4,Ggyy4,Ggyz4],[Ggzx4,Ggzy4,Ggzz4]])

Tgx1 = 3
Tgx2 = 4
Tgx3 = 3
Tgx4 = 1
Tgy1 = 7
Tgy2 = 3
Tgy3 = 9
Tgy4 = 2
Tgz1 = 5
Tgz2 = 1
Tgz3 = 4
Tgz4= 5
Tg1 = np.array([[Tgx1,Tgy1,Tgz1]])
Tg2 = np.array([[Tgx2,Tgy2,Tgz2]])
Tg3 = np.array([[Tgx3,Tgy3,Tgz3]])
Tg4 = np.array([[Tgx4,Tgy4,Tgz4]])
dTempG1 = 5
dTempG2 = 7
dTempG3 = 3
dTempG4 = 1

wb1 = np.zeros([len(t)-2,3])
wb2 = np.zeros([len(t)-2,3])
wb3 = np.zeros([len(t)-2,3])
wb4 = np.zeros([len(t)-2,3])
i = 0

for x in wb1:
    wb1[i,:] = (np.matmul((I+Sg1+Mg1),w[i,:].transpose()).transpose()+bg1+np.matmul(Gg1,acal1[i,:].transpose()).transpose()+Tg1*dTempG1)/Kg1
    wb2[i,:] = (np.matmul((I+Sg2+Mg2), w[i,:].transpose()).transpose()+bg2+np.matmul(Gg2,acal2[i,:].transpose()).transpose()+Tg2*dTempG2)/Kg2
    wb3[i,:] = (np.matmul((I+Sg3+Mg3),w[i,:].transpose()).transpose()+bg3+np.matmul(Gg3,acal3[i,:].transpose()).transpose()+Tg3*dTempG3)/Kg3
    wb4[i,:] = (np.matmul((I+Sg4+Mg4),w[i,:].transpose()).transpose()+bg4+np.matmul(Gg4,acal4[i,:].transpose()).transpose()+Tg4*dTempG4)/Kg4
    i += 1

#for AWGN see: https://stackoverflow.com/questions/14058340/adding-noise-to-a-signal-in-python
wb = (wb1+wb2+wb3+wb4)/4    
wb_avg_db = 10 * np.log10(np.mean(wb))
nw_avg_watts = 10 ** ((wb_avg_db - SNRg) / 10)  
nw = np.random.normal(0, np.sqrt(nw_avg_watts), [len(wb),3])

wbn1 = wb1 + nw            
wbn2 = wb2 + nw         
wbn3 = wb3 + nw         
wbn4 = wb4 + nw            

wcal1 = np.zeros([len(t)-2,3])
wcal2 = np.zeros([len(t)-2,3])
wcal3 = np.zeros([len(t)-2,3])
wcal4 = np.zeros([len(t)-2,3])
i = 0
for x in wbn1: 
    wcal1[i,:] = (np.matmul(np.linalg.inv(I+Sg1+Mg1),(Kg1*wbn1[i,:]-bg1-np.matmul(Gg1,acal1[i,:].transpose()).transpose()-Tg1*dTempG1).transpose())).transpose()
    wcal2[i,:] = (np.matmul(np.linalg.inv(I+Sg2+Mg2),(Kg2*wbn2[i,:]-bg2-np.matmul(Gg2,acal2[i,:].transpose()).transpose()-Tg2*dTempG2).transpose())).transpose()
    wcal3[i,:] = (np.matmul(np.linalg.inv(I+Sg3+Mg3),(Kg3*wbn3[i,:]-bg3-np.matmul(Gg3,acal3[i,:].transpose()).transpose()-Tg3*dTempG3).transpose())).transpose()
    wcal4[i,:] = (np.matmul(np.linalg.inv(I+Sg4+Mg4),(Kg4*wbn4[i,:]-bg4-np.matmul(Gg4,acal4[i,:].transpose()).transpose()-Tg4*dTempG4).transpose())).transpose()
    i += 1

#
















#


w_est = np.zeros([len(t)-2,3])
i = 0
for x in w_est:     
    w_est[i,:] =  (wcal1[i,:]+ wcal2[i,:]+ wcal3[i,:]+ wcal4[i,:])/4;
    i += 1

q_initial = np.array([q[0,:]])
qp = q_initial
q_est = np.zeros([len(t)-2,4])
q_est_norm = np.zeros([len(t)-2,4])
i = 0
for x in q_est:  
    wc = np.hstack((np.array([[0]]),np.array([w_est[i,:]])))
    q_est_prime = 0.5 * q_product(qp , wc).transpose()
    q_est[i,:] = qp + q_est_prime*delta_t
    q_est_norm[i,:] = q_est[i,:]/np.linalg.norm(q_est[i,:])   # normalization
    qp = np.array([q_est[i,:]])
    i += 1

dw_est = np.array([np.diff(w_est[:,0]),np.diff(w_est[:,1]),np.diff(w_est[:,2])]).transpose()
alpha_est = np.hstack((np.divide(dw_est[:,0:1],dt[0:-2]), np.divide(dw_est[:,1:2],dt[0:-2]), np.divide(dw_est[:,2:3],dt[0:-2])))

p1_est = q_prod_mat(q_prod_mat(q_conj_mat(q_est_norm),np.hstack((np.array([[0]]),p1b))),q_est_norm)
p1_est = p1_est[:,1:4]
p2_est = q_prod_mat(q_prod_mat(q_conj_mat(q_est_norm),np.hstack((np.array([[0]]),p2b))),q_est_norm)
p2_est = p2_est[:,1:4]
p3_est = q_prod_mat(q_prod_mat(q_conj_mat(q_est_norm),np.hstack((np.array([[0]]),p3b))),q_est_norm)
p3_est = p3_est[:,1:4]
p4_est = q_prod_mat(q_prod_mat(q_conj_mat(q_est_norm),np.hstack((np.array([[0]]),p4b))),q_est_norm)
p4_est = p4_est[:,1:4]

pp_est = q_prod_mat(q_prod_mat(q_conj_mat(q_est_norm),np.array([[0,r,0,h]])),q_est_norm)
pp_est = pp_est[:,1:4]

pp_est = np.hstack((np.array([pp_est[:,0]]).transpose(),-np.array([pp_est[:,1]]).transpose(),np.array([pp_est[:,2]]).transpose()))

r1p_est = p1_est - pp_est
r2p_est = p2_est - pp_est
r3p_est = p3_est - pp_est
r4p_est = p4_est - pp_est


a1p_est = np.zeros([len(t)-3,3])
a2p_est = np.zeros([len(t)-3,3])
a3p_est = np.zeros([len(t)-3,3])
a4p_est = np.zeros([len(t)-3,3])
i = 0

for x in a1p_est: 
    a1p_est[i,:] = acal1[i,:] - np.cross(alpha_est[i,:],r1p_est[i,:]) - np.cross(w_est[i,:],np.cross(w_est[i,:],r1p_est[i,:]))
    a2p_est[i,:] = acal2[i,:] - np.cross(alpha_est[i,:],r2p_est[i,:]) - np.cross(w_est[i,:],np.cross(w_est[i,:],r2p_est[i,:]))
    a3p_est[i,:] = acal3[i,:] - np.cross(alpha_est[i,:],r3p_est[i,:]) - np.cross(w_est[i,:],np.cross(w_est[i,:],r3p_est[i,:]))
    a4p_est[i,:] = acal4[i,:] - np.cross(alpha_est[i,:],r4p_est[i,:]) - np.cross(w_est[i,:],np.cross(w_est[i,:],r4p_est[i,:]))
    i += 1

a_est= (a1p_est+ a2p_est+ a3p_est+ a4p_est)/4

# vx = (p(2,1)-p(1,1))/delta_t;    % vx = 0
# vy = (p(2,2)-p(1,2))/delta_t;    % vy = w1*r
# vz = (p(2,3)-p(1,3))/delta_t;    % vz = 0

vy = w1*r                
v_initial = np.array([[0,vy,0]])    
vp = v_initial

v_est = np.zeros([len(t)-3,3])
i = 0
for x in a1p_est: 
    v_est[i,:] = vp + a_est[i,:]*delta_t 
    vp = np.array([v_est[i,:]])
    i += 1

p_initial = np.array([p[0,:]])
pp = p_initial
p_est = np.zeros([len(t)-3,3])
i = 0

for x in a1p_est: 
    p_est[i,:] = pp + v_est[i,:]*delta_t
    pp = np.array([p_est[i,:]])
    i += 1


#












#

# plotting quaternion
fig, (ax1,ax2,ax3,ax4) = plt.subplots(4,1,sharex=True)
ax1.plot(t[0:-1], q[0:-1,0],label='q1')
ax1.plot(t[0:-2], q_est[:,0],label='q1_est')
ax1.legend() 
ax2.plot(t[0:-1], q[0:-1,1],label='q2')
ax2.plot(t[0:-2], q_est[:,1],label='q2_est')
ax2.legend() 
ax3.plot(t[0:-1], q[0:-1,2],label='q3')
ax3.plot(t[0:-2], q_est[:,2],label='q3_est')
ax3.legend() 
ax4.plot(t[0:-1], q[0:-1,3],label='q4')
ax4.plot(t[0:-2], q_est[:,3],label='q4_est')
ax4.legend()  
fig.text(0.5, 0.04, 'Time [sec]', ha='center')
#fig.text(0.04, 0.5, 'Amplitude', va='center', rotation='vertical')

# plotting position
fig, (ax1,ax2,ax3) = plt.subplots(3,1,sharex=True)
ax1.plot(t[0:-3], p[0:-3,0],label='p1')
ax1.plot(t[0:-3], p_est[:,0],label='p1_est')
ax1.legend() 
ax2.plot(t[0:-3], p[0:-3,1],label='p2')
ax2.plot(t[0:-3], p_est[:,1],label='p2_est')
ax2.legend() 
ax3.plot(t[0:-3], p[0:-3,2],label='p3')
ax3.plot(t[0:-3], p_est[:,2],label='p3_est')
ax3.legend()    
fig.text(0.5, 0.04, 'Time [sec]', ha='center')
#fig.text(0.04, 0.5, 'Amplitude', va='center', rotation='vertical')

# plotting angular velocity
fig, (ax1,ax2,ax3) = plt.subplots(3,1,sharex=True)
ax1.plot(t[0:-1], w[:,0],label='w1')
ax1.plot(t[0:-2], w_est[:,0],label='w1_est')
ax1.legend() 
ax2.plot(t[0:-1], w[:,1],label='w2')
ax2.plot(t[0:-2], w_est[:,1],label='w2_est')
ax2.legend() 
ax3.plot(t[0:-1], w[:,2],label='w3')
ax3.plot(t[0:-2], w_est[:,2],label='w3_est')
ax3.legend()   
fig.text(0.5, 0.04, 'Time [sec]', ha='center')
#fig.text(0.04, 0.5, 'Amplitude', va='center', rotation='vertical')

# plotting linear velocity
fig, (ax1,ax2,ax3) = plt.subplots(3,1,sharex=True)
ax1.plot(t[0:-1], v[:,0],label='v1')
ax1.plot(t[0:-3], v_est[:,0],label='v1_est')
ax1.legend()
ax2.plot(t[0:-1], v[:,1],label='v2')
ax2.plot(t[0:-3], v_est[:,1],label='v2_est')
ax2.legend()
ax3.plot(t[0:-1], v[:,2],label='v3')
ax3.plot(t[0:-3], v_est[:,2],label='v3_est')
ax3.legend()   
fig.text(0.5, 0.04, 'Time [sec]', ha='center')
#fig.text(0.04, 0.5, 'Amplitude', va='center', rotation='vertical')

# plotting linear acceleration
fig, (ax1,ax2,ax3) = plt.subplots(3,1,sharex=True)
ax1.plot(t[0:-2], a[:,0],label='a1')
ax1.plot(t[0:-3], a_est[:,0],label='a1_est')
ax1.legend() 
ax2.plot(t[0:-2], a[:,1],label='a2')
ax2.plot(t[0:-3], a_est[:,1],label='a2_est')
ax2.legend() 
ax3.plot(t[0:-2], a[:,2],label='a3')
ax3.plot(t[0:-3], a_est[:,2],label='a3_est')
ax3.legend()   
fig.text(0.5, 0.04, 'Time [sec]', ha='center')
#fig.text(0.04, 0.5, 'Amplitude', va='center', rotation='vertical')

#animation of moving body (p_est and q_est)
ca = np.arange(0,2*m.pi/0.01,dtype=np.float)*0.01
cr = r
cx = cr*np.cos(ca)
cy = cr*np.sin(ca)

plt.figure()
plt.plot(cx,cy,'b')
plt.xlim(-r-1, r+1)
plt.ylim(-r-1, r+1)
plt.axis('scaled')

L = 0.5
E = 0.2

meu_est = 2*np.arccos(q_est[:,0])

i = 0
points = 20 #number of points you want to draw
for x in range(points):
    print(i)
    # for plotting the x-axis
    x1 = p_est[i,0]+(L*np.cos(meu_est[i])) 
    y1 = p_est[i,1]+(L*np.sin(meu_est[i]))
    # for printing 'x' after the x-axis end
    x1e = p_est[i,0]+((L+E)*np.cos(meu_est[i])) 
    y1e = p_est[i,1]+((L+E)*np.sin(meu_est[i])) 
    # for plotting the y-axis
    x2 = p_est[i,0]+(L*np.cos(meu_est[i]+m.pi/2))
    y2 = p_est[i,1]+(L*np.sin(meu_est[i]+m.pi/2))
    # for printing 'x' after the x-axis end
    x2e = p_est[i,0]+((L+E)*np.cos(meu_est[i]+m.pi/2)) 
    y2e = p_est[i,1]+((L+E)*np.sin(meu_est[i]+m.pi/2))
    
    plt.plot([p_est[i,0], x1],[p_est[i,1], y1],'k')      # draw x-axis of the body at point p(i,2) and p(i,2), length L, and angle meu(i)
    plt.plot([p_est[i,0], x2],[p_est[i,1], y2],'k')      # draw y-axis of the body at point p(i,2) and p(i,2), length L, and angle meu(i)+90
    plt.text(x1e,y1e,'x')                 # print 'x' after the x-axis end
    plt.text(x2e,y2e,'y')                # print 'y' after the y-axis end
    plt.plot(p_est[i,0],p_est[i,1],'ro')                # plot a point at p(i,2) and p(i,2)
    plt.pause(0.05)
    i += int(np.ceil(len(p_est)/points))

# Wait until all figures are closed
plt.show()


