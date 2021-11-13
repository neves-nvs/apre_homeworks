import math
import numpy as np
from scipy.stats import multivariate_normal

def probXnCk(x_n, c_mean, sigma):
    res = 1/(2*math.pi)
    res *= math.exp( (-1/2) * np.transpose(np.subtract(x_n, c_mean)) * sigma * np.subtract(x_n, c_mean)) 
    return res

# Given Data
sigma_1 = np.matrix("1 0; 0 1")
sigma_2 = np.matrix("2 0; 0 2")
Sigma_n = [sigma_1, sigma_2]

x1 = np.matrix([[2], 
                [4]])
x2 = np.matrix([[-1],
                [-4]])
x3 = np.matrix([[-1], 
                [2]])
x4 = np.matrix([[4], 
                [0]])
Xn = [x1, x2, x3, x4]

Pi_n = [7/10, 3/10]

print(" ____________________________________")
print("|_____________ E-Step _______________|")
print()

print("---------- P( Xn | Ck = 1 ) ----------")
XnCk = [[ 0, 0, 0, 0], [ 0, 0, 0, 0,]]
for s, sigma in enumerate(Sigma_n):
    for i, x in enumerate(Xn):
        XnCk[s][i] = multivariate_normal(x, Xn[s], sigma)
        print(f"x{i+1} - cluster {s+1}: {XnCk[s][i]}")
print()

# print("---------- P( Ck = 1 | Xn ) ----------")
# for s, cluster in enumerate(XnCk):
#     for i, x in enumerate(cluster):
#         XnCk[s][i] *= Pi_n[s]
#         print(f"x{i+1} - cluster {s+1}: {XnCk[s][i]}")
# print()

# print("---------- P( Xn ) -------------------")

# # P_Xn = [sum(x) for x in np.transpose(XnCk)]
# P_Xn = [0]*4
# x_n = np.transpose(XnCk)
# for i, n in enumerate(x_n):
#     P_Xn[i] = sum( x_n[i] )
#     print(f"P(x{i+1}) = {P_Xn[i]}")
# print()

# print("--------- Phi( Cnk ) -----------------")
# Phi_nk = XnCk
# for n, cluster in enumerate(Phi_nk):
#     for i, x in enumerate(cluster):
#         Phi_nk[n][i] /= P_Xn[i]
#         print(f"φ{i+1}{n+1} = {Phi_nk[n][i]}")
# print()

# print(" ____________________________________")
# print("|_____________ M-Step _______________|")
# print()

# print("----------- N k ----------------------")
# N_k = [ sum(x) for x in Phi_nk]
# for i, el in enumerate(N_k):
#     print(f"N{i+1} = {el}")
# print()

# print("----------- Mean Values --------------")
# U_k = [0, 0]
# for c in range(2):
#     sum_phi = np.array([[0], [0]])
#     for i, phi in enumerate(Phi_nk[c]):
#         # print(np.array([phi_nk]), Xn[indx])
#         phi_times_x = np.dot(phi, Xn[i])
#         sum_phi = np.add(sum_phi, phi_times_x)
#     U_k[c] = (1/N_k[c]) * sum_phi
#     print(f"New μ{c+1} = {U_k[c]}")
# print()

# print("----------- Cohvariance Matrix -------")
# new_sigma_k = np.array([ np.matrix('0 0; 0 0'), np.matrix('0 0; 0 0')])

# for c, sigma in enumerate(new_sigma_k):
#     for i, phi in enumerate(Phi_nk[c]):
#         tmp = np.subtract(Xn[i], U_k[c])
#         tmp = np.matmul(tmp, np.transpose(tmp)) 
#         tmp = np.dot(phi, tmp)
#         new_sigma_k[c] = np.add(tmp, new_sigma_k[c])
    
#     new_sigma_k[c] = np.dot( (1/N_k[c]), new_sigma_k[c] )
    
#     print(f"New Σ{c+1} = {new_sigma_k[c][0]}{new_sigma_k[c][1]}")
# print()

# print("----------- New Mixing Parameter -------")
# new_Pi_n = [0, 0]

# for c in range(2):
#     new_Pi_n[c] = N_k[c]/len(Xn)
#     print(f"New π{c+1} = {new_Pi_n[c]}")


#1 - multivariate gausion para desenhar a elipse

#2 - valor é x, cluster estão afastados/proximos baseado no valor

#3 - vc dimension aproximation (count the number of parameters)