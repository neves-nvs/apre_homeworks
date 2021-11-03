import numpy as np
from math import sqrt, dist

# train values
x1 = [1, 1, 1, 0]
x2 = [1, 1, 1, 5]
x3 = [1, 0, 2, 4]
x4 = [1, 1, 2, 3]
x5 = [1, 2, 0, 7]
x6 = [1, 1, 1, 1]
x7 = [1, 2, 0, 2]
x8 = [1, 0, 2, 9]
# test values
x9 = [1, 2, 0, 0]
x10= [1, 1, 2, 1]

X = np.array([x1, x2, x3, x4, x5, x6, x7, x8])
Z = np.array([1, 3, 2, 0, 6, 4, 5, 7])

def weights():
    XT = np.transpose(X)
    return np.matmul(np.matmul(np.linalg.inv(np.matmul(XT,X)),XT),Z)

def euc_distance(x):
    sum = 0
    for val in X:
        sum += val**2
    sum = sqrt(sum)

    return sum

def function(x):
    w = weights()
    res = 0

    for j in range(0,3):
        res += weights[j] * euc_distance(x)**j

def main():
    res_x3 = function(x3)
    print("Test subjet {}: {}".format(x3, res_x3))
    res_x4 = function(x4)
    print("Test subjet {}: {}".format(x4, res_x4))


def ola(y1,y2,y3):
    w=weights()
    distance = sqrt(y1**2+y2**2+y3**2)
    pot = w[0] + w[1]*y1 + w[2]*y2**2 + w[3]*y3**3
    no_pot = w[0] + w[1]*y1 + w[2]*y2 + w[3]*y3
    dist = w[0] + w[1]*distance + w[2]*distance**2 + w[3]*distance**3
    dist_no_pot = w[0] + w[1]*distance + w[2]*distance + w[3]*distance

    return(pot, no_pot, dist, dist_no_pot)


