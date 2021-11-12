import numpy as np
import math as math
import pandas as pd
import sys


data = pd.read_csv('data_dup.dat', sep='/=', header = None, skipinitialspace=False, names =['value','name'])
df = data.copy()

# Split value column into arrays by receiver and satellite information
# split df (nums_data = numerical data ONLY)

nums_data = df['value']
pd.to_numeric(nums_data).astype('float64')

# array values: pi, c (speed of light), R (radius of earth), s (length of sidereal day)

rec = nums_data[0:4].to_numpy()

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

#deg, rad to be vectors.
#hard coded constants for testing, will need to be read from data.dat

pi =2*np.arccos(0)
c = 2.997924580000000000E+08  #SoL
R = 6.367444500000000000E+06  #radius of earth
s = 8.616408999999999651E+04  #sidereal day
p = s/2  #period

#testing stuff

B12 = np.array([40,45, 55.0, 1, 111, 50, 58.0, -1, 1372.00], dtype = float)

#functional definitions

# def sat_locs(s_i, t):
#     u = np.array([s_i[0], s_i[1], s_i[2]])
#     v = np.array([s_i[3], s_i[4], s_i[5]])
#     x = (R + s_i[7]) * (-1 * u * np.sin(2 * pi * t / p + s_i[8]) + v * np.cos(2 * pi * t / p + s_i[8]))
#     return x
#
# def deg2rad(deg): #this nested form is the only way to get the correct number, s/(360*60^2) yields a wildly different answer (+/-.0004)
#
#     [d,m,s,sign] = [deg[0],deg[1],deg[2],deg[3]]
#     Lat_d2r = 2 * pi * sign * (d + (m + s/60)/60)/360
#     [d,m,s,sign] = [deg[4],deg[5],deg[6],deg[7]]
#     Long_d2r = 2 * pi * sign * (d + (m + s / 60) / 60) / 360
#     d2r = [Lat_d2r,Long_d2r, deg[8]]
#     return d2r
#
# def rad2cart(rad):
#
#     x = (R + rad[2])*np.cos(rad[0])*np.cos(rad[1])
#     y = (R + rad[2])*np.cos(rad[0])*np.sin(rad[1])
#     z = (R + rad[2])*np.sin(rad[0])
#     cart = [x,y,z]
#     return cart
#
# def deg2cart(deg):  #if you want to do it all at once.
#
#     [dt, mt, st, sign] = [deg[0], deg[1], deg[2], deg[3]]
#     Lat_d2r = 2 * pi * sign * (dt + (mt + st / 60) / 60) / 360
#     [dg, mg, sg, sign] = [deg[4], deg[5], deg[6], deg[7]]
#     Long_d2r = 2 * pi * sign * (dg + (mg + sg / 60) / 60) / 360
#     d2r = [Lat_d2r, Long_d2r, deg[8]]
#     x = (R + d2r[2]) * np.cos(d2r[0]) * np.cos(d2r[1])
#     y = (R + d2r[2]) * np.cos(d2r[0]) * np.sin(d2r[1])
#     z = (R + d2r[2]) * np.sin(d2r[0])
#     cart = np.array([x, y, z])
#     return cart
#
# def cart2rad(xyz):
#
#     psi = np.arctan2(xyz[2], np.sqrt(xyz[0] ** 2 + xyz[1] ** 2))
#     lam = np.arctan2(xyz[1], xyz[0])
#     h = np.sqrt(xyz[0] ** 2 + xyz[1] ** 2 + xyz[2] ** 2) - R
#     rad = [psi,lam,h]
#     return rad
#
# def rad2deg(rad):
#     degs = np.array([[1,2,3],[1,2,3]], dtype=float) #spent way too long trying to figure out why i kept getting a integer...always declare your variable type robert.
#     r = [0,1]
#     signs = [0,0]
#     if rad[0] < 0: #theres a way to make this look way better didn't want to work with nested loops
#         signs[0] = -1
#     else:
#         signs[0] = 1
#     if rad[1] < 0:
#         signs[1] = -1
#     else:
#         signs[1] = 1
#     for n in r:
#         init = rad[n] * ((360 / 2) / pi) * signs[n]
#         split = math.modf(init)
#         degs[n,0] = split[1]
#         inter= math.modf(split[0]*60)
#         degs[n,1] = inter[1]
#         inter2 = inter[0]*60
#         degs[n,2] = inter2
#
#     deg = [degs[0,0], degs[0,1], degs[0,2], signs[0], degs[1,0], degs[1,1],float(degs[1,2]), signs[1], rad[2]]
#     return deg
#
def rotation_offset(x,t_r): #this will compute x_v at t_v
    a = 2*pi*t_r/s
    cos = np.cos(a)
    sin = np.sin(a)
    offset = np.array([ [cos,-sin,0] , [sin,cos,0] , [0,0,1] ])
    x_off = np.dot(offset,x)
    return x_off

def horiz_check(x_i, s_i_list, t_s):  #cartesian coords, we can run this function recursively for sets of satellites and singular x with some for loops
    truth_list = []
    n = 0
    while n < 23:
        s_i = s_i_list[n]
        # u = np.array([s_i[0], s_i[1], s_i[2]])
        # v = np.array([s_i[3], s_i[4], s_i[5]])
        # s = (R+s_i[7])*(-1*u*np.sin(2*pi*t_s/p + s_i[8]) + v*np.cos(2*pi*t_s/p + s_i[8]))
        s = sat_locs(s_i,t_s)
        f = x_i.dot(s)
        g = x_i.dot(x_i)
        if f>g:
            s_up = 1  #true
        else:
            s_up = 0  #false
        truth_list.append(s_up)
        n = n+1
    return truth_list #boolean

def sat_time(x_v_i,t_v, s_i_list):  #first attempt at a newtons method code - also worth mentioning that t_v will be read in
    t_s_list = []
    errtime = 1  #a lot of ugly declarations
    n = 0
    t_0 = t_v
    t = [t_0, t_v]
    while n < 23:
        s_i = s_i_list[n]
        u = np.array([s_i[0],s_i[1],s_i[2]])
        v = np.array([s_i[3],s_i[4],s_i[5]])
        h = s_i[7]
        phase = s_i[8]

        # sat_0 = (R+h)*(-1*u*np.sin(2*pi*t[0]/p + phase) + v*np.cos(2*pi*t[0]/p + phase))   #x_s at t_v
        sat_0 = sat_locs(s_i, t_0)
        dif_0 = (sat_0 - x_v_i)

        t[0] = t_v - np.linalg.norm(dif_0)/c
        errmax = .01/c

        while errtime >= errmax:
            sat = sat_locs(s_i, t[0])
            dif = (sat - x_v_i)
            f = np.linalg.norm(dif)**2-c**2*(t_v - t[0])**2
            fpr = (4*pi*(R + h))*np.dot(dif,sat) + 2*(c**2)*(t_v - t[0])
            t[1] = t[0] - f/fpr
            errtime = np.absolute(t[1]-t[0])
            t[0] = t[1]
        t_s_list.append(t[0])
        n = n+1
    return t_s_list

def above_index(list,item):
    start_at = -1
    locs = []
    while True:
        try:
            loc = list.index(item,start_at+1)
        except ValueError:
            break
        else:
            locs.append(loc)
            start_at = loc
    return locs


#testing




def writeout(t_s_list, above_list):
    sat_exp = []
    # sat_total = np.array([],[],[])
    n_w = 0
    len_w = len(above_list)
    while n_w < len_w:
        s_x = sat_locs(sats[above_list[n_w]],t_s_list[above_list[n_w]])
        x = s_x[0]
        y = s_x[1]
        z = s_x[2]
        sat_exp = [above_list[n_w], t_s_list[above_list[n_w]], x, y, z]
        # sys.stdout.write("{} {} {} {} {}".format(sat_exp[0],sat_exp[1],sat_exp[2],sat_exp[3],sat_exp[4]))
        # for i in range(0,len(sat_exp)):
        #     print(sat_exp[i], end=" "),
        # sat_total.append(sat_exp)
        n_w = n_w + 1
        print()
    if n_w == len_w:
        x = 0
    return sat_exp
#
# xp = writeout(t_s,above)
#
# indeces = xp #read in the standard output - a list of satellites in satndard form as listed above where s_i is a throuple, we'll have to store a variable for that
#                 #                                                                                                  throuple and assign a np.array to it
#
# def Jacobian(s_i_list, t_i_list, x_k):
#     J = np.array([],[],[])
#     k = 0
#     j = 0
#     while j <= 2:
#         while k <= 2:
#             s_i = sat_locs(s_i_list[j], t_i_list[j])
#             s_i_1 = sat_locs(s_i_list[j+1], t_i_list[j+1])
#             J[j, k] = (s_i[k]-x_k[k])/np.linalg.norm(s_i[k]-x_k[k])-(s_i_1[k]-x_k[k])/np.linalg.norm(s_i_1[k]-x_k[k])
#             k = k + 1
#         j = j + 1
#     return J
#
# def Func(s_i_list, t_i_list, x_k):
#     F = np.array([],[],[])
#     j = 0
#     k = 0
#     while j <= 2:
#         while k <= 2:
#             s_i = sat_locs(s_i_list[j], t_i_list[j])
#             s_i_1 = sat_locs(s_i_list[j + 1], t_i_list[j + 1])
#             F[j,k] = np.linalg.norm(s_i_1[k]-x_k[k]) - np.linalg.norm(s_i[k]-x_k[k]) - c*(t_i_list[j]-t_i_list[j+1])
#             k = k + 1
#         j = j + 1
#     return F
#
# def LU(A, b):
#     N = len(A)
#     L = np.zeros(shape=(N,N))
#     U = np.zeros(shape=(N,N))
#     x = np.zeros(shape=(N,1))
#     y = np.zeros(shape=(N,1))
#     for i in range(N):
#
#         for j in range(i, N): #
#             sum = 0
#             for k in range(i):
#                 sum = sum + L[i,k]*U[k,j]
#             U[i,j] = A[i,j] - sum
#         for j in range(i,N):
#             if i == j:
#                 L[i,i] = 1
#             else:
#                 sum = 0
#                 for k in range(i):
#                     sum = sum + L[j,k]*U[k,i]
#                 L[j,i] = (A[j,i]-sum)/U[i,i]
#     while i <= N:
#         if i == 0:
#             y[0,0] = b[0,0]
#             i = i + 1
#         else:
#             sum = 0
#             for k in range(i):
#                 sum = sum + y[i-1,0]*L[i,i-0]
#                 y[i,0] = b[i,0] - sum
#             i = i + 1
#     while i >= 0:
#         if i == N:
#             x[N,0] = y[N,0]/U[N,N]
#             i = i - 1
#         else:
#             sum = 0
#             for k in range(i+1,N):
#                 sum = sum + U[i,k]*x[k,0]
#             x[i,0] = (y[i,0]- sum)/U[i,i]
#     return x
#
# def Newt(s_i_list,t_i_list,x_0):
#     x_k = x_0
#     s = np.ones(shape = (3,1))
#     while np.linalg.norm(s) >= .0001:
#         J = Jacobian(s_i_list, t_i_list, x_k)
#         F = Func(s_i_list, t_i_list, x_k)
#         s = LU(J,-F)
#         x_k[0] = x_k[0] + s
#     return x_k
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

# def Jacobian(s_i_list, t_i_list, x_k):
#     J = np.zeros(shape=(3,3))
#     k = 0
#     j = 0
#     while j <= 2:
#         while k <= 2:
#             s_i = sat_locs(sats[s_i_list[j]], t_i_list[j])
#             s_i_1 = sat_locs(sats[s_i_list[j+1]], t_i_list[j+1])
#             J[j, k] = (s_i[k]-x_k[k])/np.linalg.norm(s_i[k]-x_k[k])-(s_i_1[k]-x_k[k])/np.linalg.norm(s_i_1[k]-x_k[k])
#             k = k + 1
#         j = j + 1
#     return J


def Jacobian(s_i_list, t_i_list, x_k):
    J = np.zeros(shape=(3,3), dtype=float)
    s0 = sat_locs(sats[s_i_list[0]], t_i_list[0])
    s1 = sat_locs(sats[s_i_list[1]], t_i_list[1])
    s2 = sat_locs(sats[s_i_list[2]], t_i_list[2])
    s3 = sat_locs(sats[s_i_list[3]], t_i_list[3])
    for k in range(0,3):
        J[0, k] = ((s0[k] - x_k[k]) / np.linalg.norm(s0 - x_k)) - ((s1[k] - x_k[k]) / np.linalg.norm(s1 - x_k))
        J[1, k] = ((s1[k] - x_k[k]) / np.linalg.norm(s1 - x_k)) - ((s2[k] - x_k[k]) / np.linalg.norm(s2 - x_k))
        J[2, k] = ((s2[k] - x_k[k]) / np.linalg.norm(s2 - x_k)) - ((s3[k] - x_k[k]) / np.linalg.norm(s3 - x_k))
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



    # for i in range(1,N-1):
    #     for j in range(0, N-1):
    #         A_int[i,j] =A_int[i,j] - A_int[i,0]/A_int[0,0]
    #         L[i,i] = 1






def Newt(s_i_list,t_i_list,x_0):
    x_k = np.array([0,0,0], dtype=float)
    x_k[0] = x_0[0]
    x_k[1] = x_0[1]
    x_k[2] = x_0[2]
    x = [x_k,x_0]
    err=1
    errmax = .001
    s_k = np.ones(shape=(3,1), dtype=float)
    while err >= errmax:
        J = Jacobian(s_i_list, t_i_list, x[0])
        F = Func(s_i_list, t_i_list, x[0])
        s_k = LU(J,-F)
        x[1] = x[0] + s_k
        err = np.linalg.norm(s_k)
        x[0] = x[1]

    return x_k

x_v_c = deg2cart(B12)  #B12 here will be replaced by a read-in x_v from vehicle.log
t_v = 12123.0  #will be read in
x_v_t = rotation_offset(x_v_c, t_v)

n = 0

t_s = sat_time(x_v_t,t_v,sats)
s_ab = horiz_check(x_v_t,sats,t_v)

above = above_index(s_ab,1)

x_b12 = Newt(above,t_s, deg2cart(B12))
x_b12rad = cart2rad(x_b12)
x_b12deg = rad2deg(x_b12rad)
# print(deg2cart(B12))
# print(J)
# print(F)
# s_k = LU(J,-F)
print(*x_b12deg)












