'''
#Implement LM algorithm only using basic python
#Author:Leo Ma
#For csmath2019 assignment4,ZheJiang University
#Date:2019.04.28
'''
import numpy as np
import matplotlib.pyplot as plt

import cv2
import glob
import os

# input data, whose shape is (num_data,1)
# data_input=np.array([[0.25, 0.5, 1, 1.5, 2, 3, 4, 6, 8]]).T
# data_output=np.array([[19.21, 18.15, 15.36, 14.10, 12.89, 9.32, 7.45, 5.24, 3.01]]).T


tao = 10 ** -3
threshold_stop = 10 ** -20
threshold_step = 10 ** -20
threshold_residual = 10 ** -20
residual_memory = []


# construct a user function
def my_Func(params, input_data):
    x1 = params[0, 0]
    x6 = params[5, 0]
    x3 = params[2, 0]
    x4 = params[3, 0]
    x2 = params[1, 0]
    x5 = params[4, 0]
    x9 = params[8, 0]
    x12 = params[11, 0]
    x7 = params[6, 0]
    x8 = params[7, 0]
    x10 = params[9, 0]
    x11 = params[10, 0]
    x13 = params[12, 0]
    x14 = params[13, 0]
    xw, yw = input_data[:, 0], input_data[:, 1]
    out_x =  1.0/(xw+x8 + yw*x11 + x14) * (xw*(x1*x6+x3*x7+x4*x8) + yw*(x1*x9+x3*x10+x4*x11) + x1*x12+x3*x13+x4*x14)
    out_y =  1.0/(xw+x8 + yw*x11 + x14) * (xw*(x2*x7+x5*x8) + yw*(x2*x10+x5*x11) + x2*x13+x5*x14)
    out_xy = np.zeros(input_data.shape)
    out_xy[:, 0] = out_x
    out_xy[:, 1] = out_y
    return out_xy


# generating the input_data and output_data,whose shape both is (num_data,1)
def generate_data(params, num_data):
    x = np.array((np.linspace(0, 10, num_data), np.linspace(0, 10, num_data))).reshape(num_data, 2)  # 产生包含噪声的数据
    mid, sigma = 0, 5
    noise = np.random.normal(mid, sigma, num_data *2).reshape(num_data, 2)
    y = my_Func(params, x) + noise
    return x, y


# calculating the derive of pointed parameter,whose shape is (num_data,1)
def cal_deriv(params, input_data, param_index):
    params1 = params.copy()
    params2 = params.copy()
    params1[param_index, 0] += 0.000001
    params2[param_index, 0] -= 0.000001
    data_est_output1 = my_Func(params1, input_data)
    data_est_output2 = my_Func(params2, input_data)
    return (data_est_output1 - data_est_output2) / 0.000002

def deriv_f1_x1(params, input_data):
    x6 = params[5, 0]
    x8 = params[7, 0]
    x9 = params[8, 0]
    x11 = params[10, 0]
    x12 = params[11, 0]
    x14 = params[13, 0]
    xw, yw = input_data[:, 0], input_data[:, 1]
    t = 1.0/(xw+x8 + yw*x11 + 1) * (x6*xw + yw*x9 + x12)
    return 1.0/(xw*x8 + yw*x11 + x14) * (x6*xw + yw*x9 + x12)

def deriv_f1_x2(params, input_data):
    xw, yw = input_data[:, 0], input_data[:, 1]
    return np.zeros(xw.shape)

def deriv_f1_x3(params, input_data):
    x7 = params[6, 0]
    x8 = params[7, 0]
    x10 = params[9, 0]
    x11 = params[10, 0]
    x13 = params[12, 0]
    x14 = params[13, 0]
    xw, yw = input_data[:, 0], input_data[:, 1]
    return 1.0/(xw*x8 + yw*x11 + x14) * (x7*xw + yw*x10 + x13)

def deriv_f1_x4(params, input_data):
    xw, yw = input_data[:, 0], input_data[:, 1]
    return np.ones(xw.shape)

def deriv_f1_x5(params, input_data):
    xw, yw = input_data[:, 0], input_data[:, 1]
    return np.zeros(xw.shape)

def deriv_f1_x6(params, input_data):
    x1 = params[0, 0]
    x8 = params[7, 0]
    x11 = params[10, 0]
    x14 = params[13, 0]
    xw, yw = input_data[:, 0], input_data[:, 1]
    return 1.0/(xw*x8 + yw*x11 + x14) * (x1*xw)

def deriv_f1_x7(params, input_data):
    x3 = params[2, 0]
    x8 = params[7, 0]
    x11 = params[10, 0]
    x14 = params[13, 0]
    xw, yw = input_data[:, 0], input_data[:, 1]
    return 1.0/(xw*x8 + yw*x11 + x14) * (x3*xw)

def deriv_f1_x8(params, input_data):
    x1 = params[0, 0]
    x6 = params[5, 0]
    x3 = params[2, 0]
    x7 = params[6, 0]
    x4 = params[3, 0]
    x8 = params[7, 0]
    x9 = params[8, 0]
    x10 = params[9, 0]
    x11 = params[10, 0]
    x12 = params[11, 0]
    x13 = params[12, 0]
    x14 = params[13, 0]
    xw, yw = input_data[:, 0], input_data[:, 1]
    return 1.0/((xw*x8 + yw*x11 + x14)**2) * (x4*xw*(xw*x8 + yw*x11 + x14) - xw*(xw*(x1*x6+x3*x7+x4*x8) + yw*(x1*x9+x3*x10+x4*x11) + x1*x12+x3*x13+x4*x14))

def deriv_f1_x9(params, input_data):
    x1 = params[0, 0]
    x8 = params[7, 0]
    x11 = params[10, 0]
    x14 = params[13, 0]
    xw, yw = input_data[:, 0], input_data[:, 1]
    return 1.0/(xw*x8 + yw*x11 + x14) * yw*x1

def deriv_f1_x10(params, input_data):
    x3 = params[2, 0]
    x8 = params[7, 0]
    x11 = params[10, 0]
    x14 = params[13, 0]
    xw, yw = input_data[:, 0], input_data[:, 1]
    return 1.0/(xw*x8 + yw*x11 + x14) * yw*x3

def deriv_f1_x11(params, input_data):
    x1 = params[0, 0]
    x6 = params[5, 0]
    x3 = params[2, 0]
    x7 = params[6, 0]
    x4 = params[3, 0]
    x8 = params[7, 0]
    x9 = params[8, 0]
    x10 = params[9, 0]
    x11 = params[10, 0]
    x12 = params[11, 0]
    x13 = params[12, 0]
    x14 = params[13, 0]
    xw, yw = input_data[:, 0], input_data[:, 1]
    return 1.0/((xw*x8 + yw*x11 + x14)**2) * (x11*yw*(xw*x8 + yw*x11 + x14) - yw*(xw*(x1*x6+x3*x7+x4*x8) + yw*(x1*x9+x3*x10+x4*x11) + x1*x12+x3*x13+x4*x14))

def deriv_f1_x12(params, input_data):
    x1 = params[0, 0]
    x8 = params[7, 0]
    x11 = params[10, 0]
    xw, yw = input_data[:, 0], input_data[:, 1]
    x14 = params[13, 0]
    return 1.0/(xw*x8 + yw*x11 + x14) * x1

def deriv_f1_x13(params, input_data):
    x3 = params[2, 0]
    x8 = params[7, 0]
    x11 = params[10, 0]
    x14 = params[13, 0]
    xw, yw = input_data[:, 0], input_data[:, 1]
    return 1.0/(xw*x8 + yw*x11 + x14) * x3

def deriv_f1_x14(params, input_data):
    x1 = params[0, 0]
    x6 = params[5, 0]
    x3 = params[2, 0]
    x7 = params[6, 0]
    x4 = params[3, 0]
    x8 = params[7, 0]
    x9 = params[8, 0]
    x10 = params[9, 0]
    x11 = params[10, 0]
    x12 = params[11, 0]
    x13 = params[12, 0]
    x14 = params[13, 0]
    xw, yw = input_data[:, 0], input_data[:, 1]
    return 1.0 / ((xw * x8 + yw * x11 + x14) ** 2) * (x4 * (xw * x8 + yw * x11 + x14) - (
                xw * (x1 * x6 + x3 * x7 + x4 * x8) + yw * (
                    x1 * x9 + x3 * x10 + x4 * x11) + x1 * x12 + x3 * x13 + x4 * x14))

def deriv_f1_x15(params, input_data):
    xw, yw = input_data[:, 0], input_data[:, 1]
    return np.zeros(xw.shape)

def deriv_f1_x16(params, input_data):
    xw, yw = input_data[:, 0], input_data[:, 1]
    return np.zeros(xw.shape)

def deriv_f1_x17(params, input_data):
    xw, yw = input_data[:, 0], input_data[:, 1]
    return np.zeros(xw.shape)


def deriv_f2_x1(params, input_data):
    xw, yw = input_data[:, 0], input_data[:, 1]
    return np.zeros(xw.shape)

def deriv_f2_x2(params, input_data):
    x7 = params[6, 0]
    x10 = params[9, 0]
    x13 = params[12, 0]
    x8 = params[7, 0]
    x11 = params[10, 0]
    x14 = params[13, 0]
    xw, yw = input_data[:, 0], input_data[:, 1]
    return 1.0/(xw*x8 + yw*x11 + x14) * (xw*x7+yw*x10+x13)

def deriv_f2_x3(params, input_data):
    xw, yw = input_data[:, 0], input_data[:, 1]
    return np.zeros(xw.shape)

def deriv_f2_x4(params, input_data):
    xw, yw = input_data[:, 0], input_data[:, 1]
    return np.zeros(xw.shape)

def deriv_f2_x5(params, input_data):
    xw, yw = input_data[:, 0], input_data[:, 1]
    return np.ones(xw.shape)

def deriv_f2_x6(params, input_data):
    xw, yw = input_data[:, 0], input_data[:, 1]
    return np.zeros(xw.shape)

def deriv_f2_x7(params, input_data):
    x2 = params[1, 0]
    x8 = params[7, 0]
    x11 = params[10, 0]
    x14 = params[13, 0]
    xw, yw = input_data[:, 0], input_data[:, 1]
    return 1.0/(xw*x8 + yw*x11 + x14) * xw*x2

def deriv_f2_x8(params, input_data):
    x2 = params[1, 0]
    x5 = params[4, 0]
    x7 = params[6, 0]
    x8 = params[7, 0]
    x10 = params[9, 0]
    x11 = params[10, 0]
    x13 = params[12, 0]
    x14 = params[13, 0]
    xw, yw = input_data[:, 0], input_data[:, 1]
    return 1.0/((xw*x8 + yw*x11 + x14)**2) * (x5*xw*(xw*x8 + yw*x11 + x14) - xw*(xw*(x2*x7+x5*x8) + yw*(x2*x10+x5*x11) + x2*x13+x5*x14))

def deriv_f2_x9(params, input_data):
    xw, yw = input_data[:, 0], input_data[:, 1]
    return np.zeros(xw.shape)

def deriv_f2_x10(params, input_data):
    x2 = params[1, 0]
    x8 = params[7, 0]
    x11 = params[10, 0]
    x14 = params[13, 0]
    xw, yw = input_data[:, 0], input_data[:, 1]
    return 1.0/(xw*x8 + yw*x11 + x14) * yw*x2

def deriv_f2_x11(params, input_data):
    x2 = params[1, 0]
    x5 = params[4, 0]
    x7 = params[6, 0]
    x8 = params[7, 0]
    x10 = params[9, 0]
    x11 = params[10, 0]
    x13 = params[12, 0]
    x14 = params[13, 0]
    xw, yw = input_data[:, 0], input_data[:, 1]
    return 1.0/((xw*x8 + yw*x11 + x14)**2) * (x5*yw*(xw*x8 + yw*x11 + x14) - yw*(xw*(x2*x7+x5*x8) + yw*(x2*x10+x5*x11) + x2*x13+x5*x14))

def deriv_f2_x12(params, input_data):
    xw, yw = input_data[:, 0], input_data[:, 1]
    return np.zeros(xw.shape)

def deriv_f2_x13(params, input_data):
    x2 = params[1, 0]
    x8 = params[7, 0]
    x11 = params[10, 0]
    x14 = params[13, 0]
    xw, yw = input_data[:, 0], input_data[:, 1]
    return 1.0/(xw*x8 + yw*x11 + x14) * x2

def deriv_f2_x14(params, input_data):
    x2 = params[1, 0]
    x5 = params[4, 0]
    x7 = params[6, 0]
    x8 = params[7, 0]
    x10 = params[9, 0]
    x11 = params[10, 0]
    x13 = params[12, 0]
    x14 = params[13, 0]
    xw, yw = input_data[:, 0], input_data[:, 1]
    return 1.0/((xw*x8 + yw*x11 + x14)**2) * (x5*(xw*x8 + yw*x11 + x14) - (xw*(x2*x7+x5*x8) + yw*(x2*x10+x5*x11) + x2*x13+x5*x14))

def deriv_f2_x15(params, input_data):
    xw, yw = input_data[:, 0], input_data[:, 1]
    return np.zeros(xw.shape)

def deriv_f2_x16(params, input_data):
    xw, yw = input_data[:, 0], input_data[:, 1]
    return np.zeros(xw.shape)

def deriv_f2_x17(params, input_data):
    xw, yw = input_data[:, 0], input_data[:, 1]
    return np.zeros(xw.shape)



def myderiv_fun():
    f1_fun_list = []
    f2_fun_list = []
    fun_list = []
    f1_fun_list.append(deriv_f1_x1)
    f1_fun_list.append(deriv_f1_x2)
    f1_fun_list.append(deriv_f1_x3)
    f1_fun_list.append(deriv_f1_x4)
    f1_fun_list.append(deriv_f1_x5)
    f1_fun_list.append(deriv_f1_x6)
    f1_fun_list.append(deriv_f1_x7)
    f1_fun_list.append(deriv_f1_x8)
    f1_fun_list.append(deriv_f1_x9)
    f1_fun_list.append(deriv_f1_x10)
    f1_fun_list.append(deriv_f1_x11)
    f1_fun_list.append(deriv_f1_x12)
    f1_fun_list.append(deriv_f1_x13)
    f1_fun_list.append(deriv_f1_x14)
    f1_fun_list.append(deriv_f1_x15)
    f1_fun_list.append(deriv_f1_x16)
    f1_fun_list.append(deriv_f1_x17)

    f2_fun_list.append(deriv_f2_x1)
    f2_fun_list.append(deriv_f2_x2)
    f2_fun_list.append(deriv_f2_x3)
    f2_fun_list.append(deriv_f2_x4)
    f2_fun_list.append(deriv_f2_x5)
    f2_fun_list.append(deriv_f2_x6)
    f2_fun_list.append(deriv_f2_x7)
    f2_fun_list.append(deriv_f2_x8)
    f2_fun_list.append(deriv_f2_x9)
    f2_fun_list.append(deriv_f2_x10)
    f2_fun_list.append(deriv_f2_x11)
    f2_fun_list.append(deriv_f2_x12)
    f2_fun_list.append(deriv_f2_x13)
    f2_fun_list.append(deriv_f2_x14)
    f2_fun_list.append(deriv_f2_x15)
    f2_fun_list.append(deriv_f2_x16)
    f2_fun_list.append(deriv_f2_x17)

    fun_list.append(f1_fun_list)
    fun_list.append(f2_fun_list)
    return fun_list

def deriv_function(params, input_data, param_index, fun_index):
    all_deriv_fun = myderiv_fun()

    params1 = params.copy()
    data_est_output = all_deriv_fun[fun_index][param_index](params1, input_data)
    return data_est_output

def lagrange_contraint(params):
    x1 = params[0, 0]
    x6 = params[5, 0]
    x3 = params[2, 0]
    x4 = params[3, 0]
    x2 = params[1, 0]
    x5 = params[4, 0]
    x9 = params[8, 0]
    x12 = params[11, 0]
    x7 = params[6, 0]
    x8 = params[7, 0]
    x10 = params[9, 0]
    x11 = params[10, 0]
    x13 = params[12, 0]
    x14 = params[13, 0]
    x15 = params[14, 0]
    x16 = params[15, 0]
    x17 = params[16, 0]
    lagrange_contraint_fun = []
    contraint_fun = -1*x15*(x6*x9+x7*x10+x8*x11)-x16*(x6**2+x7**2+x8**2-1)-x17*(x9**2+x10**2+x11**2-1)
    partial_fun_x1 = 0
    partial_fun_x2 = 0
    partial_fun_x3 = 0
    partial_fun_x4 = 0
    partial_fun_x5 = 0
    partial_fun_x6 = -x15*x9-2*x16
    partial_fun_x7 = -x15*x10-2*x16
    partial_fun_x8 = -x15*x11-2*x16
    partial_fun_x9 = -x15*x6 -2*x17
    partial_fun_x10 = -x15*x7 -2*x17
    partial_fun_x11 = -x15*x8 -2*x17
    partial_fun_x12 = 0
    partial_fun_x13 = 0
    partial_fun_x14 = 0
    partial_fun_x15 = -1*(x6*x9+x7*x10+x8*x11)
    partial_fun_x16 = -1*(x6**2+x7**2+x8**2-1)
    partial_fun_x17 = -1*(x9**2+x10**2+x11**2-1)
    lagrange_contraint_fun.append(partial_fun_x1)
    lagrange_contraint_fun.append(partial_fun_x2)
    lagrange_contraint_fun.append(partial_fun_x3)
    lagrange_contraint_fun.append(partial_fun_x4)
    lagrange_contraint_fun.append(partial_fun_x5)
    lagrange_contraint_fun.append(partial_fun_x6)
    lagrange_contraint_fun.append(partial_fun_x7)
    lagrange_contraint_fun.append(partial_fun_x8)
    lagrange_contraint_fun.append(partial_fun_x9)
    lagrange_contraint_fun.append(partial_fun_x10)
    lagrange_contraint_fun.append(partial_fun_x11)
    lagrange_contraint_fun.append(partial_fun_x12)
    lagrange_contraint_fun.append(partial_fun_x13)
    lagrange_contraint_fun.append(partial_fun_x14)
    lagrange_contraint_fun.append(partial_fun_x15)
    lagrange_contraint_fun.append(partial_fun_x16)
    lagrange_contraint_fun.append(partial_fun_x17)
    return lagrange_contraint_fun

# calculating jacobian matrix,whose shape is (num_data,num_params)
def cal_Jacobian(params, input_data):
    num_params = np.shape(params)[0]
    num_data = np.shape(input_data)[0]
    J = np.zeros((num_data, num_params))
    residual_u_x, residual_v_y = cal_residual_u_x_v_y(params, input_data)
    for i in range(num_params):
        lagrange_constraint = lagrange_contraint(params)[i]
        deriv_for_param_i = residual_u_x * deriv_function(params, input_data, i, 0) + residual_v_y * deriv_function(params, input_data, i, 1) + lagrange_constraint
        J[:, i] = list(deriv_for_param_i)
    return J


# calculating residual, whose shape is (num_data,1)
def cal_residual(params, input_data, output_data):
    x1 = params[0, 0]
    x6 = params[5, 0]
    x3 = params[2, 0]
    x4 = params[3, 0]
    x2 = params[1, 0]
    x5 = params[4, 0]
    x9 = params[8, 0]
    x12 = params[11, 0]
    x7 = params[6, 0]
    x8 = params[7, 0]
    x10 = params[9, 0]
    x11 = params[10, 0]
    x13 = params[12, 0]
    x14 = params[13, 0]
    x15 = params[14, 0]
    x16 = params[15, 0]
    x17 = params[16, 0]
    contraint_fun = -1 * x15 * (x6 * x9 + x7 * x10 + x8 * x11) - x16 * (x6 ** 2 + x7 ** 2 + x8 ** 2 - 1) - x17 * (
                x9 ** 2 + x10 ** 2 + x11 ** 2 - 1)
    data_est_output = my_Func(params, input_data)
    residual = np.linalg.norm(output_data - data_est_output, axis=1).reshape(input_data.shape[0], 1) + contraint_fun
    return residual


def cal_residual_u_x_v_y(params, input_data):
    data_est_output = my_Func(params, input_data)
    u_estimation = data_est_output[:, 0]
    v_estimation = data_est_output[:, 1]
    real_u = input_data[:, 0]
    real_v = input_data[:, 1]
    residual_u_x = u_estimation - real_u
    residual_v_y = v_estimation - real_v
    return residual_u_x, residual_v_y

'''    
#calculating Hessian matrix, whose shape is (num_params,num_params)
def cal_Hessian_LM(Jacobian,u,num_params):
    H = Jacobian.T.dot(Jacobian) + u*np.eye(num_params)
    return H

#calculating g, whose shape is (num_params,1)
def cal_g(Jacobian,residual):
    g = Jacobian.T.dot(residual)
    return g

#calculating s,whose shape is (num_params,1)
def cal_step(Hessian_LM,g):
    s = Hessian_LM.I.dot(g)
    return s

'''


# get the init u, using equation u=tao*max(Aii)
def get_init_u(A, tao):
    m = np.shape(A)[0]
    Aii = []
    for i in range(0, m):
        Aii.append(A[i, i])
    u = tao * max(Aii)
    return u


# LM algorithm
def LM(num_iter, params, input_data, output_data):
    num_params = np.shape(params)[0]  # the number of params
    k = 0  # set the init iter count is 0
    # calculating the init residual
    residual = cal_residual(params, input_data, output_data)
    # calculating the init Jocobian matrix
    Jacobian = cal_Jacobian(params, input_data)

    A = Jacobian.T.dot(Jacobian)  # calculating the init A
    g = Jacobian.T.dot(residual)  # calculating the init gradient g
    stop = (np.linalg.norm(g, ord=np.inf) <= threshold_stop)  # set the init stop
    u = get_init_u(A, tao)  # set the init u
    v = 2  # set the init v=2

    while ((not stop) and (k < num_iter)):
        k += 1
        while (1):
            Hessian_LM = A + u * np.eye(num_params)  # calculating Hessian matrix in LM
            step = np.linalg.inv(Hessian_LM).dot(g)  # calculating the update step
            if (np.linalg.norm(step) <= threshold_step):
                stop = True
            else:
                new_params = params + step  # update params using step
                new_residual = cal_residual(new_params, input_data, output_data)  # get new residual using new params
                rou = (np.linalg.norm(residual) ** 2 - np.linalg.norm(new_residual) ** 2) / (step.T.dot(u * step + g))
                if rou > 0:
                    params = new_params
                    residual = new_residual
                    residual_memory.append(np.linalg.norm(residual) ** 2)
                    # print (np.linalg.norm(new_residual)**2)
                    Jacobian = cal_Jacobian(params, input_data)  # recalculating Jacobian matrix with new params
                    A = Jacobian.T.dot(Jacobian)  # recalculating A
                    g = Jacobian.T.dot(residual)  # recalculating gradient g
                    stop = (np.linalg.norm(g, ord=np.inf) <= threshold_stop) or (
                                np.linalg.norm(residual) ** 2 <= threshold_residual)
                    u = u * max(1 / 3, 1 - (2 * rou - 1) ** 3)
                    v = 2
                else:
                    u = u * v
                    v = 2 * v
            if (rou > 0 or stop):
                break;

    return params


def main():
    # set the true params for generate_data() function
    params = np.zeros((17, 1))
    params[0, 0] = 10.0
    params[1, 0] = 0.8
    params[2, 0] = 10.0
    params[3, 0] = 0.8
    params[4, 0] = 10.0
    params[5, 0] = 0.8
    params[6, 0] = 10.0
    params[7, 0] = 0.8
    params[8, 0] = 10.0
    params[9, 0] = 0.8
    params[10, 0] = 10.0
    params[11, 0] = 0.8
    params[12, 0] = 0.8
    params[13, 0] = 0.8
    params[14, 0] = 0
    params[15, 0] = 0
    params[16, 0] = 0
    num_data = 100  # set the data number
    data_input, data_output = generate_data(params, num_data)  # generate data as requested


    # 设置寻找亚像素角点的参数，采用的停止准则是最大循环次数30和最大误差容限0.001
    criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)

    # 获取标定板角点的位置
    objp = np.zeros((6 * 8, 3), np.float32)
    objp[:, :2] = np.mgrid[0:8, 0:6].T.reshape(-1, 2)  # 将世界坐标系建在标定板上，所有点的Z坐标全部为0，所以只需要赋值x和y

    obj_points = []  # 存储3D点
    img_points = []  # 存储2D点

    root_path = "/home/mry/Documents/CVPR-ppt/Assignment1/Camera_A/Mode1/"
    images = os.listdir(root_path)
    image = images[10:11]
    i = 0;
    for fname in image:
        print(fname)
        img = cv2.imread(os.path.join(root_path, fname))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        size = gray.shape[::-1]
        ret, corners = cv2.findChessboardCorners(gray, (8, 6), None)
        # print(corners)

        if ret:

            obj_points.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)  # 在原角点的基础上寻找亚像素角点
            # print(corners2)
            if [corners2]:
                img_points.append(corners2)
            else:
                img_points.append(corners)

            cv2.drawChessboardCorners(img, (8, 6), corners, ret)  # 记住，OpenCV的绘制函数一般无返回值
            i += 1;
            cv2.imwrite('conimg' + str(i) + '.jpg', img)
            cv2.waitKey(1500)

    print(len(img_points))
    cv2.destroyAllWindows()
    num_data, _, _ = img_points[0].shape
    data_input = obj_points[0][:, :2]
    data_output = img_points[0].reshape(num_data, 2)



    # set the init params for LM algorithm
    params[0, 0] = 1497
    params[1, 0] = 1485
    params[2, 0] = 0.1
    params[3, 0] = 1296
    params[4, 0] = 1000
    params[5, 0] = 0
    params[6, 0] = 0
    params[7, 0] = 0
    params[8, 0] = 0
    params[9, 0] = 0
    params[10, 0] = 0
    params[11, 0] = -33.86
    params[12, 0] = 17.07
    params[13, 0] = 255.45
    params[14, 0] = 0
    params[15, 0] = 0
    params[16, 0] = 0
    # using LM algorithm estimate params
    num_iter = 500  # the number of iteration
    est_params = LM(num_iter, params, data_input, data_output)
    print(est_params)

    xw, yw = data_input[:, 0], data_input[:, 1]
    #plt.scatter(data_output[:, 0], data_output[:, 1], color='r')
    scale_lambda = est_params[7, 0]*xw +est_params[10, 0]*yw + est_params[13, 0]
    # 老子画个图看看状况
    estimation = my_Func(est_params, data_input)
    plt.scatter(estimation[:, 0], estimation[:, 1], color='b')
    x = np.arange(0, 100) * 0.1  # 生成0-10的共100个数据，然后设置间距为0.1
    #plt.plot(x, a_est * np.exp(b_est * x), 'r', lw=1.0)
    plt.savefig("result_LM.png")
    plt.show()

    plt.scatter(data_output[:, 0], data_output[:, 1], color='r')
    plt.show()
    print('ground_truth')

    plt.plot(residual_memory)
    plt.savefig("error-iter.png")
    plt.show()


if __name__ == '__main__':
    main()
