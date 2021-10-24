import numpy as np
from math import exp, pi, sqrt
import sys
    
Y1 = np.array([0.6,0.1,0.2,0.1,0.3,-0.1,-0.3,0.2,0.4,-0.2])
Y2 = np.array(["A","B","A","C","B","C","C","B","A","C"])
Y3 = np.array([0.2,-0.1,-0.1,0.8,0.1,0.2,-0.1,0.5,-0.4,0.4])
Y4 = np.array([0.4,-0.4,0.2,0.8,0.3,-0.2,0.2,0.6,-0.7,0.3])
Class = np.array([0,0,0,0,1,1,1,1,1,1])

my10=np.average(Y1[:4])
sy10=np.std(Y1[:4],ddof=1)
my11=np.average(Y1[4:])
sy11=np.std(Y1[4:],ddof=1)
my1=np.average(Y1)
sy1=np.std(Y1,ddof=1)

my30=np.average(Y3[:4])
sy30=np.std(Y3[:4],ddof=1)
my31=np.average(Y3[4:])
sy31=np.std(Y3[4:],ddof=1)
my3=np.average(Y3)
sy3=np.std(Y3,ddof=1)

my40=np.average(Y4[:4])
sy40=np.std(Y4[:4],ddof=1)
my41=np.average(Y4[4:])
sy41=np.std(Y4[4:],ddof=1)
my4=np.average(Y4)
sy4=np.std(Y4,ddof=1)

Ey3y40=np.cov(np.array([Y3[:4],Y4[:4]]))
Ey3y41=np.cov(np.array([Y3[4:],Y4[4:]]))
Ey3y4=np.cov(np.array([Y3,Y4]))

def prob_c(c):
    return np.count_nonzero(Class==c)/Class.shape[0]

def prob_normal(x,m,s):
    return 1/(sqrt(2*pi)*s) * exp(-1/(2*s**2)*(x-m)**2)

def prob_disc(x,cl):
    if(cl==0):
        return np.count_nonzero(Y2[:4]==x)/4
    elif(cl==1):
        return np.count_nonzero(Y2[4:]==x)/6
    else:
        return np.count_nonzero(Y2==x)/Y2.shape[0]

def prob_normal_bivar(x,m,E):
    return 1/(2*pi*sqrt(np.linalg.det(E)))*exp(-0.5*np.matmul(np.matmul((x-m).transpose(),np.linalg.inv(E)),(x-m)))

def prob_class(y1,y2,y3,y4,denominadores=True):
    py1_c0 = prob_normal(y1,my10,sy10)
    py2_c0 = prob_disc(y2,0)
    py3y4_c0 = prob_normal_bivar(np.array([y3,y4]),np.array([my30,my40]),Ey3y40)
    py1_c1 = prob_normal(y1,my11,sy11)
    py2_c1 = prob_disc(y2,1)
    py3y4_c1 = prob_normal_bivar(np.array([y3,y4]),np.array([my31,my41]),Ey3y41)
    pc0 = prob_c(0)
    pc1 = prob_c(1)
    py1 = prob_normal(y1,my1,sy1)
    py2 = prob_disc(y2,0)
    py3y4 = prob_normal_bivar(np.array([y3,y4]),np.array([my3,my4]),Ey3y4)

    py1y2y3y3y4c0=py1_c0*py2_c0*py3y4_c0*pc0
    py1y2y3y3y4c1=py1_c1*py2_c1*py3y4_c1*pc1

    if(denominadores):
        return py1y2y3y3y4c0/(py1y2y3y3y4c0+py1y2y3y3y4c1), py1y2y3y3y4c1/(py1y2y3y3y4c0+py1y2y3y3y4c1)

    else:
        return py1y2y3y3y4c0/(py1*py2*py3y4), py1y2y3y3y4c1/(py1*py2*py3y4)

def delta(y1,y2,y3,y4):
    pipinho, teorico = prob_class(y1,y2,y3,y4)

    print("teorico:",teorico)
    print("pipinho:", pipinho)
    print(pipinho-teorico)

def confusion(threshold=0.5,denominadores=True):
    tp = []
    fp = []
    tn = []
    fn = []
    
    for i in range(0,10):
        p0,p1=prob_class(Y1[i],Y2[i],Y3[i],Y4[i],denominadores)
        print("x{}: p0={:.2f} p1={:.2f}".format(1+i,p0,p1))
        if(p1>threshold):
            if(Class[i]==1):
                tn.append("x"+str(1+i))
            else:
                fn.append("x"+str(1+i))
        else:
            if(Class[i]==0):
                tp.append("x"+str(1+i))
            else:
                fp.append("x"+str(1+i))

    print(tp,fp,tn,fn)
    print("accuracy:"+str(((len(tp)+len(tn))/10)))
