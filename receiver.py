import numpy as np
import math as math
import pandas as pd
import sys

data = pd.read_csv('data.dat', sep='/=', header = None, skipinitialspace=False, names =['value','name'], engine="python")
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
    len_w = len(above_list)
    while n_w < len_w:
        s_x = sat_locs(sats[above_list[n_w]], t_s_list[above_list[n_w]])
        x = s_x[0]
        y = s_x[1]
        z = s_x[2]
        sat_exp = [above_list[n_w], t_s_list[above_list[n_w]], x, y, z]
        sys.stdout.write("{} {} {} {} {}".format(sat_exp[0], sat_exp[1], sat_exp[2], sat_exp[3], sat_exp[4]))
        n_w = n_w + 1
        print()
    if n_w == len_w:
        x = 0
    return sat_exp

def Jacobian(s_i_list, t_i_list, x_k):
    J = np.zeros(shape=(3,3))
    k = 0
    j = 0
    while j <= 2:
        while k <= 2:
            s_i = sat_locs(sats[s_i_list[j]], t_i_list[j])
            s_i_1 = sat_locs(sats[s_i_list[j+1]], t_i_list[j+1])
            J[j, k] = (s_i[k]-x_k[k])/np.linalg.norm(s_i[k]-x_k[k])-(s_i_1[k]-x_k[k])/np.linalg.norm(s_i_1[k]-x_k[k])
            k = k + 1
        j = j + 1
    return J

def Func(s_i_list, t_i_list, x_k):
    F = np.zeros(shape=(3,1))
    j = 0
    k = 0
    while j <= 2:
        s_i = sat_locs(sats[s_i_list[j]], t_i_list[j])
        s_i_1 = sat_locs(sats[s_i_list[j + 1]], t_i_list[j + 1])
        s_i_1_r= cart2rad(s_i_1)
        s_i_1_d = rad2deg(s_i_1_r)
        F[j,0] = np.linalg.norm(s_i_1-x_k) - np.linalg.norm(s_i-x_k) - c*(t_i_list[j]-t_i_list[j+1])
        j = j + 1
    return F

# def LU(A, b):
#     N = len(A)
#     L = np.zeros(shape=(N, N))
#     U = np.zeros(shape=(N, N))
#     x = np.zeros(shape=(N, 1))
#     y = np.zeros(shape=(N, 1))
#     for i in range(N):
#
#         for j in range(i, N):  #
#             sum = 0
#             for k in range(i):
#                 sum = sum + L[i, k] * U[k, j]
#             U[i, j] = A[i, j] - sum
#         for j in range(i, N):
#             if i == j:
#                 L[i, i] = 1
#             else:
#                 sum = 0
#                 for k in range(i):
#                     sum = sum + L[j, k] * U[k, i]
#                 L[j, i] = (A[j, i] - sum) / U[i, i]
#
#     while i <= N-1:
#         if i == 0:
#             y[0, 0] = b[0, 0]
#             i = i + 1
#         else:
#             sum = 0
#             for k in range(i):
#                 sum = sum + y[i - 1, 0] * L[i, i - 1]
#                 y[i, 0] = b[i, 0] - sum
#             i = i + 1
#     while i >= 0:
#         if i == N-1:
#             x[i, 0] = y[i, 0] / U[i, i]
#             i = i - 1
#         else:
#             sum = 0
#             for k in range(i , N-1):
#                 sum = sum + U[i, k] * x[k, 0]
#             x[i, 0] = (y[i, 0] - sum) / U[i, i]
#     return x


def LU(A,b_):
    A_int = np.array([[A[0,0],A[0,1],A[0,2]],[A[1,0],A[1,1],A[1,2]],[A[2,0],A[2,1],A[2,2]]], dtype=float)
    L = np.zeros(shape=(3,3), dtype=float)
    U = np.zeros(shape=(3,3), dtype=float)
    n = len(A)
    for i in range(n):

        #Upper
        for k in range(i, n):
            sum = 0
            for j in range(0,i):
                sum += float((L[i][j]*U[j][k]))
            U[i][k] = float(A_int[i][k] - sum)

        #Lowe
        for k in range(i, n):
            if (i == k):
                L[i][i] = 1
            else:
                sum = 0
                for j in range(i):
                    sum += float(L[k][j]*U[j][i])
                L[k][i] = ((A_int[k][i] - sum)/U[i][i])
    y = np.array([0,0,0], dtype=float)
    x = np.array([0,0,0], dtype=float)
    N = len(A)
    b = np.array([b_[0][0],b_[1][0],b_[2][0]],dtype=float)
    y[0] = float(b[0])
    y[1] = float(b[1] - y[0]*L[1,0])
    y[2] = float(b[2] - (y[1]*L[2,1]+y[0]*L[2,0]))
    x[2] = y[2]/U[2,2]
    x[1] = (y[1]-x[2]*U[1,2])/U[1,1]
    int = x[1]*U[0,1]-x[2]*U[0,2]
    x[0] = (y[0]-x[1]*U[0,1]-x[2]*U[0,2])/U[0,0]
    return x

def Newt(s_i_list,t_i_list,x_0):
    x_k = np.array([0,0,0], dtype=float)
    x_k[0] = x_0[0]
    x_k[1] = x_0[1]
    x_k[2] = x_0[2]
    x = [x_k,x_0]
    err=1
    errmax = .00001
    s_k = np.ones(shape=(3,1), dtype=float)
    while err >= errmax:
        J = Jacobian(s_i_list, t_i_list, x[0])
        F = Func(s_i_list, t_i_list, x[0])
        s_k = LU(J,-F)
        x[1] = x[0] + s_k
        err = np.linalg.norm(s_k)
        x[0] = x[1]

    return x_k


lines = sys.stdin.readlines()
log = open("receiver.log", "w+")
log.write("Log: Robert Caldwell, Emily Toney. \n Input:\n")
for line in lines:
    log.write("{}\n".format(line))

for line in lines:
    lines_strip = line.rsplit()
    lines_float = []
    data = np.zeros(shape=(len(lines), len(lines_strip)))
# def receive():
#     i = 0
#     for line in lines:
#         lines_strip = line.rsplit()
#         lines_float = []
#         for n in range(0,len(lines_strip)):
#             lines_float.append( float(lines_strip[n]))
#             data[i,n] = lines_float[n]
#         i = i + 1
#     #data has ALL lines from stdin, need to run each step separately - need a way to separate
#     j=0
#     print(data)
#     leng = len(data)
#     for j in range(0,leng):
#         if data[j,0] > data[j-1,0]:
#             sat_list = []
#             t_list = []
#             for k in range(0, len(data)):
#                 sat_list.append(int(data[k, 0]))
#                 t_list.append(float(data[k, 1]))
#         x = Newt(sat_list, t_list, deg2cart(B12))
#         x_r = cart2rad(x)
#         x_d = rad2deg(x_r)
#         x_s = np.array([data[0, 2], data[0, 3], data[0, 4]])
#         t_v = np.linalg.norm(x - x_s) / c + data[0, 1]
#         sys.stdout.write(
#             "{} {} {} {} {} {} {} {} {} {}\n".format(t_v, x_d[0], x_d[1], x_d[2], x_d[3], x_d[4], x_d[5], x_d[6],
#                                                      x_d[7], x_d[8]))
#         log.write("{} {} {} {} {} {} {} {} {} {}\n".format(t_v, x_d[0], x_d[1], x_d[2], x_d[3], x_d[4], x_d[5], x_d[6],
#                                                            x_d[7], x_d[8]))
#         i = i + 1
def receive():
    i = 0
    for line in lines:
        lines_strip = line.rsplit()
        lines_float = []
        for n in range(0,len(lines_strip)):
            lines_float.append( float(lines_strip[n]))
            data[i,n] = lines_float[n]
        i = i + 1
    j=0
    leng = len(data)
    i = 0
    sat_list = []
    t_list = []
    split = 0
    t_s_x=[]
    while i < leng:
        if data[i][0] - data[i - 1][0] < 0:
            split = i
            i = i+1

        for k in range(0, len(data)):
            sat_list.append(int(data[k][0]))
            t_list.append(int(data[k][1]))

        for j in range(0, leng):

            if data[j][0] - data[j-1][0] > 0:
                j = j + 1
            else:
                x = Newt(sat_list, t_list, deg2cart(B12))
                x_r = cart2rad(x)
                x_d = rad2deg(x_r)
                x_s = np.array([data[j][2], data[j][3], data[j][4]])
                t_v = np.linalg.norm(x - x_s) / c + data[j, 1]
                sys.stdout.write(
                    "{} {} {} {} {} {} {} {} {} {}\n".format(t_v, x_d[0], x_d[1], x_d[2], x_d[3], x_d[4], x_d[5], x_d[6],
                                                             x_d[7], x_d[8]))
                log.write("{} {}".format( x_d[7], x_d[8]))
                i = i + 1
# def receive():
#     i = 0
#     for line in lines:
#         lines_strip = line.rsplit()
#         lines_float = []
#         for n in range(0, len(lines_strip)):
#             lines_float.append(float(lines_strip[n]))
#             data[i, n] = lines_float[n]
#         i = i + 1
#     j = 0
#     leng = len(data)
#     i = 0
#     sat_list = []
#     t_list = []
#     split = 0
#     t_s_x = []
#     while i < leng:
#         if data[i][0] - data[i - 1][0] < 0:
#             split = i
#             i = i + 1
#
#             for k in range(0, len(data)):
#                 sat_list.append(int(data[k][0]))
#                 t_list.append(int(data[k][1]))
#             for k in range(0, split):
#                 t_s_x[k] = t_list[split + k]
#                 print(t_s_x)
#             x = Newt(sat_list, t_list, deg2cart(B12))
#             x_r = cart2rad(x)
#             x_d = rad2deg(x_r)
#             x_s = np.array([data[j][2], data[j][3], data[j][4]])
#             t_v = np.linalg.norm(x - x_s) / c + data[j, 1]
#             sys.stdout.write(
#                 "{} {} {} {} {} {} {} {} {} {}\n".format(t_v, x_d[0], x_d[1], x_d[2], x_d[3], x_d[4], x_d[5],
#                                                          x_d[6],
#                                                          x_d[7], x_d[8]))
#             log.write("{} {}".format( x_d[7], x_d[8]))
#             i = i + 1
#         else:
#             i = i + 1



receive()
