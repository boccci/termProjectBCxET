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
sat_0 = nums_data[4:13].to_numpy()

#satellites

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


#deg, rad to be vectors.
#hard coded constants for testing, will need to be read from data.dat

pi =2*np.arccos(0)
c = 2.997924580000000000E+08  #SoL
R = 6.367444500000000000E+06  #radius of earth
s = 8.616408999999999651E+04  #sidereal day
p = s/2  #period

#sattelite hardcode

u_1 = np.array([1, 0, 0], dtype = float)
v_1 = np.array([1, .573757664363510461594, .8191520442889917986], dtype = float)
pd_1 = 4.308204499999999825E+04
hs_1 =   2.020000000000000000E+07

s = np.array ([u_1,v_1,pd_1,hs_1])
s_1 = [u_1,v_1,pd_1,hs_1]
#testing stuff
B12 = np.array([40,45, 55.0, 1, 111, 50, 58.0, -1, 1372.00], dtype = float)

#longa = np.array([gd, gm, gs, ew])


#functional definitions

def deg2rad(deg): #this nested form is the only way to get the correct number, s/(360*60^2) yields a wildly different answer (.0004)

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

def deg2cart(deg): #if you want to do it all at once.

    [d, m, s, sign] = [deg[0], deg[1], deg[2], deg[3]]
    Lat_d2r = 2 * pi * sign * (d + (m + s / 60) / 60) / 360
    [d, m, s, sign] = [deg[4], deg[5], deg[6], deg[7]]
    Long_d2r = 2 * pi * sign * (d + (m + s / 60) / 60) / 360
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
    if rad[0] < 0: #theres a way to make this look way better didnt want to work with nested loops
        signs[0] = -1
    else:
        signs[0] =  1
    if rad[1] < 0:
        signs[1] = -1
    else:
        signs[1] =  1
    for n in r:
        init = rad[n] * ((360 / (2)) / pi) * signs[n]
        split = math.modf(init)
        degs[n,0] = split[1]
        inter= math.modf(split[0]*60)
        degs[n,1] = inter[1]
        inter2 = inter[0]*60
        degs[n,2] = inter2

    deg = [degs[0,0], degs[0,1], degs[0,2], signs[0], degs[1,0], degs[1,1],float(degs[1,2]) , signs[1], rad[2]]
    return deg

def rotation_offset(x,t_r):

    a = 2*pi*t_r/s
    offset = np.matrix ([ [np.cos(a),-np.sin(a),0] , [np.sin(a),np.cos(a),0] , [0,0,1] ])
    x_off = offset*x
    return x_off

def horiz_check(x, s, t_s): #cartesian coords, we can run this function recursively for sets of sattelites and singular x with some for loops

    s = (-1 * s[0] * np.sin(2 * pi * t_s / p + s[2]) + s[1] * np.cos(2 * pi * t_s / p + s[2]))
    dif = s - x
    f = 0

    for n in (0,2):
        f = f + x[n]*dif[n]
    if f>0:
        s_up = 1 #true
    else:
        s_up = 0 #false
    return s_up #boolean

def sat_time(x_v_i,t_v, s_i):  #first attempt at a newtons method code - also worth mentioning that t_v will be read in
    errtime = 0  #a lot of ugly declarations
    t_0 = t_v
    t = [t_0, t_v]
    u = np.array([s_i[0],s_i[1],s_i[2]])
    h = s_i[7]
    phase = s_i[8]
    
    sat_0 = (-1 * u * np.sin(2 * pi * t[0] / p + phase) + v * np.cos(2 * pi * t[0] / p + phase))   #x_s at t_v
    dif_0 = (sat_0 - x_v_i)

    t_0 = t_v - np.linalg.norm(dif_0)

    while errtime < .01/c:
        sat = ( -1 * u * np.sin(2 * pi * t[0] / p + phase) + v * np.cos(2 * pi * t[0] / p + phase ))
        dif = ( sat - x_v_i )
        f = np.linalg.norm(dif)-c**2*( t_v - t[0] )**2
        fpr = ( 4*pi( R+h ) ) * np.multiply(dif,sat)+2*c**2*( t_v - t[0] )
        t[1] = t_0 - f/fpr
        errtime = np.sqrt(t[1]**2-t[0]**2)
    t_s = t[1]
    return t_s

# def sat_time(x_v_i,t_v, u_s,v_s,phase_s):  #first attempt at a newtons method code
#     errtime = 0  #a lot of ugly declarations
#     t_0 = t_v
#     t = [t_0, 0]
#
#     sat_0 = (-1 * u_s * np.sin(2 * pi * t[0] / p + phase_s) + v_s * np.cos(2 * pi * t[0] / p + phase_s))
#     dif_0 = (sat_0 - x_v_i)
#
#     t_0 = t_v - np.linalg.norm(dif_0)
#     h = cart2rad(x_v_i)
#
#     while errtime < .01/c:
#         sat = ( -1 * u_s * np.sin(2 * pi * t[0] / p + phase_s) + v_s * np.cos(2 * pi * t[0] / p + phase_s ))
#         dif = ( sat - x_v_i )
#         f = np.linalg.norm(dif)-c**2*( t_v - t[0] )**2
#         fpr = ( 4*pi( R+h ) ) * np.multiply(dif,sat)+2*c**2*( t_v - t[0] )
#         t[1] = t_0 - f/fpr
#         errtime = np.sqrt(t[1]**2-t[0]**2)
#     t_s = t[1]
#     return t_s



#tests

B12r = deg2rad(B12)
B12c = rad2cart(B12r)
B12a1a1 = deg2cart(B12)
B12rev = cart2rad(B12a1a1)
B12rev2 = rad2deg(B12rev)

print(*B12)
print(*B12rev)
print(*B12rev2)

B12above = np.array([40,45, 55.0, 1, 111, 50, 58.0, -1, 13720.00], dtype = float)
B12ac = deg2cart(B12above)
print(check)
