import numpy as np
import math as math
import pandas as pd
from decimal import *

data = pd.read_csv('data_dup.dat', sep='/=', header = None, skipinitialspace=False, names =['value','name'])
df = data.copy()

# Split value column into arrays by receiver and satellite information
# split df (nums_data = numerical data ONLY)

nums_data = df['value']
pd.to_numeric(nums_data).astype('float64')

# array values: pi, c (speed of light), R (radius of earth), s (length of sidereal day)

rec = nums_data[0:4].to_numpy()
pi = 2*np.arccos(0)
c = rec[1]
R = rec[2]
s = rec[3]
p = s/2

#satellites

sat_0 = nums_data[4:13].to_numpy()
sat_1 = nums_data[13:22].to_numpy()
sat_2 = nums_data[22:31].to_numpy()
sat_3 = nums_data[31:40].to_numpy()
sat_4 = nums_data[40:49].to_numpy()
sat_5 = nums_data[49:58].to_numpy()
sat_6 = nums_data[58:67].to_numpy()
sat_7 = nums_data[67:76].to_numpy()
sat_8 = nums_data[76:85].to_numpy()
sat_9 = nums_data[85:94].to_numpy()
sat_10 = nums_data[94:103].to_numpy()
sat_11 = nums_data[103:112].to_numpy()
sat_12 = nums_data[112:121].to_numpy()
sat_13 = nums_data[121:130].to_numpy()
sat_14 = nums_data[130:139].to_numpy()
sat_15 = nums_data[139:148].to_numpy()
sat_16 = nums_data[148:157].to_numpy()
sat_17 = nums_data[157:166].to_numpy()
sat_18 = nums_data[166:175].to_numpy()
sat_19 = nums_data[175:184].to_numpy()
sat_20 = nums_data[184:193].to_numpy()
sat_21 = nums_data[193:202].to_numpy()
sat_22 = nums_data[202:211].to_numpy()
sat_23 = nums_data[211:220].to_numpy()

sats = [sat_0, sat_1, sat_2, sat_3, sat_4, sat_5, sat_6, sat_7, sat_8, sat_9, sat_10, sat_11, sat_12, sat_13, sat_14, sat_15, sat_16, sat_17, sat_18, sat_19, sat_20, sat_21, sat_22, sat_23]

#

s_ab_list = []

#testing stuff

B12 = np.array([40,45, 55.0, 1, 111, 50, 58.0, -1, 1372.00], dtype = float)

#functional definitions

def sat_locs(s_i, t):
    u = np.array([s_i[0], s_i[1], s_i[2]])
    v = np.array([s_i[3], s_i[4], s_i[5]])
    x = (R + s_i[7]) * (-1 * u * np.sin(2 * pi * t / p + s_i[8]) + v * np.cos(2 * pi * t / p + s_i[8]))
    return x

def deg2rad(deg): #this nested form is the only way to get the correct number, s/(360*60^2) yields a wildly different answer (+/-.0004)

    [d,m,s,sign] = [deg[0],deg[1],deg[2],deg[3]]
    Lat_d2r = 2 * pi * sign * (d + (m + s/60)/60)/360
    [d,m,s,sign] = [deg[4],deg[5],deg[6],deg[7]]
    Long_d2r = 2 * pi * sign * (d + (m + s / 60) / 60) / 360
    d2r = [Lat_d2r,Long_d2r, deg[8]]
    return d2r

def rad2cart(rad):

    x = (R + rad[2])*np.cos(rad[0])*np.cos(rad[1])
    y = (R + rad[2])*np.cos(rad[0])*np.sin(rad[1])
    z = (R + rad[2])*np.sin(rad[0])
    cart = [x,y,z]
    return cart

def deg2cart(deg):  #if you want to do it all at once.

    [dt, mt, st, sign] = [deg[0], deg[1], deg[2], deg[3]]
    Lat_d2r = 2 * pi * sign * (dt + (mt + st / 60) / 60) / 360
    [dg, mg, sg, sign] = [deg[4], deg[5], deg[6], deg[7]]
    Long_d2r = 2 * pi * sign * (dg + (mg + sg / 60) / 60) / 360
    d2r = [Lat_d2r, Long_d2r, deg[8]]
    x = (R + d2r[2]) * np.cos(d2r[0]) * np.cos(d2r[1])
    y = (R + d2r[2]) * np.cos(d2r[0]) * np.sin(d2r[1])
    z = (R + d2r[2]) * np.sin(d2r[0])
    cart = np.array([x, y, z])
    return cart

def cart2rad(xyz):

    psi = np.arctan2(xyz[2], np.sqrt(xyz[0] ** 2 + xyz[1] ** 2))
    lam = np.arctan2(xyz[1], xyz[0])
    h = np.sqrt(xyz[0] ** 2 + xyz[1] ** 2 + xyz[2] ** 2) - R
    rad = [psi,lam,h]
    return rad

def rad2deg(rad):
    degs = np.array([[1,2,3],[1,2,3]], dtype=float) #spent way too long trying to figure out why i kept getting a integer...always declare your variable type robert.
    r = [0,1]
    signs = [0,0]
    if rad[0] < 0: #theres a way to make this look way better didn't want to work with nested loops
        signs[0] = -1
    else:
        signs[0] = 1
    if rad[1] < 0:
        signs[1] = -1
    else:
        signs[1] = 1
    for n in r:
        init = rad[n] * ((360 / 2) / pi) * signs[n]
        split = math.modf(init)
        degs[n,0] = split[1]
        inter= math.modf(split[0]*60)
        degs[n,1] = inter[1]
        inter2 = inter[0]*60
        degs[n,2] = inter2

    deg = [degs[0,0], degs[0,1], degs[0,2], signs[0], degs[1,0], degs[1,1],float(degs[1,2]), signs[1], rad[2]]
    return deg

def rotation_offset(x,t_r): #this will compute x_v at t_v
    a = 2*pi*t_r/s
    cos = np.cos(a)
    sin = np.sin(a)
    offset = np.array([ [cos,-sin,0] , [sin,cos,0] , [0,0,1] ])
    x_off = np.dot(offset,x)
    return x_off

def writeout(t_s_list, above_list):
    sat_exp = []
    # sat_total = np.array([],[],[])
    n_w = 0
    len_w = len(above_list)-1
    while n_w <= len_w:
        s_x = sat_locs(sats[above_list[n_w]],t_s_list[above_list[n_w]])
        x = s_x[0]
        y = s_x[1]
        z = s_x[2]
        above_list[n_w]
        sat_exp = [above_list[n_w], t_s_list[above_list[n_w]], x, y, z]
        print(*sat_exp),
        # sat_total.append(sat_exp)
        n_w = n_w + 1

    return sat_exp #need a way to save the last x as an array without it telling me its an array


read_in = open(r"read_in.txt", "w+")

import subprocess #this should all be written as a single command
proc = subprocess.Popen(['python','satellite.py'],stdout=subprocess.PIPE) #will need to ask for engine and program and then input those instead
while True:
    line = proc.stdout.readline()
    l = str(line.rstrip())
    if not line:
        break
    read_in.write('{}\n'.format(l))

read_in.close()

with open("read_in.txt", 'r') as file:
    Data = []
    n = 0
    for lines in file:
        Data.append(lines.replace("\n","").replace("'","").replace('"',"").replace('b',""))
        n = n+1

print(n)

# # data_array = np.ndarray(shape=(n,3))
# #
# # print(*data_array)
#
# print(Data[0])
# data_array = np.ndarray(shape = (n,2))
#
# i=0
#
# while i <= n-1:
#     data = Data[i]
#     data_array[i, 0] = float(data.split(' ')[0])
#     data_array[i, 1] = float(data.split(' ')[1])
#     i = i+1
# i = 0

def Jacobian(s_i_list, t_i_list, x_k):
    J = np.array([],[],[])
    k = 0
    j = 0
    while j <= 2:
        while k <= 2:
            s_i = sat_locs(s_i_list[j], t_i_list[j])
            s_i_1 = sat_locs(s_i_list[j+1], t_i_list[j+1])
            J[j, k] = (s_i[k]-x_k[k])/np.linalg.norm(s_i[k]-x_k[k])-(s_i_1[k]-x_k[k])/np.linalg.norm(s_i_1[k]-x_k[k])
            k = k + 1
        j = j + 1
    return J

def Func(s_i_list, t_i_list, x_k):
    F = np.array([],[],[])
    j = 0
    k = 0
    while j <= 2:
        while k <= 2:
            s_i = sat_locs(s_i_list[j], t_i_list[j])
            s_i_1 = sat_locs(s_i_list[j + 1], t_i_list[j + 1])
            F[j,k] = np.linalg.norm(s_i_1[k]-x_k[k]) - np.linalg.norm(s_i[k]-x_k[k]) - c*(t_i_list[j]-t_i_list[j+1])
            k = k + 1
        j = j + 1
    return F

def LU(A, b):
    N = len(A)
    L = np.zeros(shape=(N,N))
    U = np.zeros(shape=(N,N))
    x = np.zeros(shape=(N,1))
    y = np.zeros(shape=(N,1))
    for i in range(N):

        for j in range(i, N): #
            sum = 0
            for k in range(i):
                sum = sum + L[i,k]*U[k,j]
            U[i,j] = A[i,j] - sum
        for j in range(i,N):
            if i == j:
                L[i,i] = 1
            else:
                sum = 0
                for k in range(i):
                    sum = sum + L[j,k]*U[k,i]
                L[j,i] = (A[j,i]-sum)/U[i,i]
    while i <= N:
        if i == 0:
            y[0,0] = b[0,0]
            i = i + 1
        else:
            sum = 0
            for k in range(i):
                sum = sum + y[i-1,0]*L[i,i-0]
                y[i,0] = b[i,0] - sum
            i = i + 1
    while i >= 0:
        if i == N:
            x[N,0] = y[N,0]/U[N,N]
            i = i - 1
        else:
            sum = 0
            for k in range(i+1,N):
                sum = sum + U[i,k]*x[k,0]
            x[i,0] = (y[i,0]- sum)/U[i,i]
    return x

def Newt(s_i_list,t_i_list,x_0):
    x_k = x_0
    s = np.array([0],[0],[0])
    while np.linalg.norm(s) >= .0001:
        J = Jacobian(s_i_list, t_i_list, x_k)
        F = Func(s_i_list, t_i_list, x_k)
        s = LU(J,-F)
        x_k[0] = x_k[0] + s
    return x_k



