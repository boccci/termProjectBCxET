import numpy as np
import math as math
import pandas as pd
import sys

data = pd.read_csv('data.dat', sep='/=', header=None, skipinitialspace=False, names=['value', 'name'], engine='python')
df = data.copy()

#
import select

# Split value column into arrays by receiver and satellite information
# split df (nums_data = numerical data ONLY)

nums_data = df['value']
pd.to_numeric(nums_data).astype('float64')

# array values: pi, c (speed of light), R (radius of earth), s (length of sidereal day)

rec = nums_data[0:4].to_numpy()
pi = 2 * np.arccos(0)
c = rec[1]
R = rec[2]
s = rec[3]
p = s / 2

# satellites

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

sats = [sat_0, sat_1, sat_2, sat_3, sat_4, sat_5, sat_6, sat_7, sat_8, sat_9, sat_10, sat_11, sat_12, sat_13, sat_14,
        sat_15, sat_16, sat_17, sat_18, sat_19, sat_20, sat_21, sat_22, sat_23]
s_ab_list = []

B12 = np.array([40, 45, 55.0, 1, 111, 50, 58.0, -1, 1372.00], dtype=float)


# functional definitions

def sat_locs(s_i, t):
    u = np.array([s_i[0], s_i[1], s_i[2]])
    v = np.array([s_i[3], s_i[4], s_i[5]])
    x = (R + s_i[7]) * (-1 * u * np.sin(2 * pi * t / p + s_i[8]) + v * np.cos(2 * pi * t / p + s_i[8]))
    return x


def deg2rad(
        deg):  # this nested form is the only way to get the correct number, s/(360*60^2) yields a wildly different answer (+/-.0004)

    [d, m, s, sign] = [deg[0], deg[1], deg[2], deg[3]]
    Lat_d2r = 2 * pi * sign * (d + (m + s / 60) / 60) / 360
    [d, m, s, sign] = [deg[4], deg[5], deg[6], deg[7]]
    Long_d2r = 2 * pi * sign * (d + (m + s / 60) / 60) / 360
    d2r = [Lat_d2r, Long_d2r, deg[8]]
    return d2r


def rad2cart(rad):
    x = (R + rad[2]) * np.cos(rad[0]) * np.cos(rad[1])
    y = (R + rad[2]) * np.cos(rad[0]) * np.sin(rad[1])
    z = (R + rad[2]) * np.sin(rad[0])
    cart = [x, y, z]
    return cart


def deg2cart(deg):  # if you want to do it all at once.

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
    rad = [psi, lam, h]
    return rad


def rad2deg(rad):
    degs = np.array([[1, 2, 3], [1, 2, 3]],
                    dtype=float)  # spent way too long trying to figure out why i kept getting a integer...always declare your variable type robert.
    r = [0, 1]
    signs = [0, 0]
    if rad[0] < 0:  # theres a way to make this look way better didn't want to work with nested loops
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
        degs[n, 0] = split[1]
        inter = math.modf(split[0] * 60)
        degs[n, 1] = inter[1]
        inter2 = inter[0] * 60
        degs[n, 2] = inter2

    deg = [degs[0, 0], degs[0, 1], degs[0, 2], signs[0], degs[1, 0], degs[1, 1], float(degs[1, 2]), signs[1], rad[2]]
    return deg


def rotation_offset(x, t_r):  # this will compute x_v at t_v
    a = 2 * pi * t_r / s
    cos = np.cos(a)
    sin = np.sin(a)
    offset = np.array([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]])
    x_off = np.dot(offset, x)
    return x_off


# def horiz_check(x_i, s_i_list, t_s):  #cartesian coords, we can run this function recursively for sets of satellites and singular x with some for loops
#     truth_list = []
#     n = 0
#     while n < 24:
#         s_i = s_i_list[n]
#         # u = np.array([s_i[0], s_i[1], s_i[2]])
#         # v = np.array([s_i[3], s_i[4], s_i[5]])
#         # s = (R+s_i[7])*(-1*u*np.sin(2*pi*t_s/p + s_i[8]) + v*np.cos(2*pi*t_s/p + s_i[8]))
#         s = sat_locs(s_i,t_s)
#         f = np.dot(x_i,s)
#         g = np.dot(x_i,x_i)
#         if f>g:
#             s_up = 1  #true
#         else:
#             s_up = 0  #false
#         truth_list.append(s_up)
#         n = n+1
#     return truth_list #boolean

def horiz_check(x_i, s_i_list,
                t_s):  # cartesian coords, we can run this function recursively for sets of satellites and singular x with some for loops
    truth_list = []
    n = 0
    while n < 24:
        s_i = s_i_list[n]
        # u = np.array([s_i[0], s_i[1], s_i[2]])
        # v = np.array([s_i[3], s_i[4], s_i[5]])
        # s = (R+s_i[7])*(-1*u*np.sin(2*pi*t_s/p + s_i[8]) + v*np.cos(2*pi*t_s/p + s_i[8]))
        s_dif = sat_locs(s_i, t_s) - x_i
        f = np.dot(x_i, s_dif)
        if f > 0:
            s_up = 1  # true
        else:
            s_up = 0  # false
        truth_list.append(s_up)
        n = n + 1
    return truth_list  # boolean


def sat_time(x_v_i, t_v,
             s_i_list):  # first attempt at a newtons method code - also worth mentioning that t_v will be read in
    t_s_list = []
    errtime = 1  # a lot of ugly declarations
    n = 0
    t_0 = t_v
    t = [t_0, t_v]
    while n < 24:
        s_i = s_i_list[n]
        h = s_i[7]
        sat_0 = sat_locs(s_i, t_0)
        dif_0 = (sat_0 - x_v_i)

        t[0] = t_v - np.linalg.norm(dif_0) / c
        errmax = .001 / c

        while errtime >= errmax:
            sat = sat_locs(s_i, t[0])
            dif = (sat - x_v_i)
            f = np.linalg.norm(dif) ** 2 - c ** 2 * (t_v - t[0]) ** 2
            fpr = (4 * pi * (R + h)) * np.dot(dif, sat) + 2 * (c ** 2) * (t_v - t[0])
            t[1] = t[0] - f / fpr
            errtime = np.absolute(t[1] - t[0])
            t[0] = t[1]
        t_s_list.append(t[0])
        n = n + 1
    return t_s_list


def above_index(list, item):
    start_at = -1
    locs = []
    while True:
        try:
            loc = list.index(item, start_at + 1)
        except ValueError:
            break
        else:
            locs.append(loc)
            start_at = loc
    return locs


# testing

vehicle = np.array([0,0,0,0,0,0,0,0,0,0])

for line in sys.stdin:
    lines_strip = line.rsplit()
    lines_float=[]
    for n in range(0,len(lines_strip)):
        lines_float[n] = float(lines_strip[n])
    x_v = np.array([[lines_float[1]],[lines_float[1]],[lines_float[1]],[lines_float[1]],[lines_float[1]],[lines_float[1]],[lines_float[1]],[lines_float[1]]])
    x_v_c = deg2cart(x_v)  # B12 here will be replaced by a read-in x_v from vehicle.log
    t_v = lines_float[0]  # will be read in
    x_v_t = rotation_offset(x_v_c, t_v)

    n = 0

    t_s = sat_time(x_v_t, t_v, sats)
    s_ab = horiz_check(x_v_t, sats, t_v)
    above = above_index(s_ab, 1)


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


      xp = writeout(t_s, above)
