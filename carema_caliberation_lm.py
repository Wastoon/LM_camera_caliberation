'''
#Implement LM algorithm only using basic python
#Author:Leo Ma
#For csmath2019 assignment4,ZheJiang University
#Date:2019.04.28
'''
import numpy as np
import matplotlib.pyplot as plt

# input data, whose shape is (num_data,1)
# data_input=np.array([[0.25, 0.5, 1, 1.5, 2, 3, 4, 6, 8]]).T
# data_output=np.array([[19.21, 18.15, 15.36, 14.10, 12.89, 9.32, 7.45, 5.24, 3.01]]).T


tao = 10 ** -3
threshold_stop = 10 ** -15
threshold_step = 10 ** -15
threshold_residual = 10 ** -15
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
    xw, yw = input_data[:, 0], input_data[:, 1]
    out_x =  1.0/(xw+x8 + yw*x11 + 1) * (xw*(x1*x6+x3*x7+x4*x8) + yw*(x1*x9+x3*x10+x4*x11) + x1*x12+x3*x13+x4)
    out_y =  1.0/(xw+x8 + yw*x11 + 1) * (xw*(x2*x7+x5*x8) + yw*(x2*x10+x5*x11) + x2*x13+x5)
    out_xy = np.zeros(input_data.shape)
    out_xy[:, 0] = out_x
    out_xy[:, 1] = out_y
    return out_xy


# generating the input_data and output_data,whose shape both is (num_data,1)
def generate_data(params, num_data):
    x = np.array(np.linspace(0, 10, num_data)).reshape(num_data//2, 2)  # 产生包含噪声的数据
    mid, sigma = 0, 5
    noise = np.random.normal(mid, sigma, num_data).reshape(num_data//2, 2)
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
    xw, yw = input_data[:, 0], input_data[:, 1]
    t = 1.0/(xw+x8 + yw*x11 + 1) * (x6*xw + yw*x9 + x12)
    return 1.0/(xw+x8 + yw*x11 + 1) * (x6*xw + yw*x9 + x12)

def deriv_f1_x2(params, input_data):
    xw, yw = input_data[:, 0], input_data[:, 1]
    return np.zeros(xw.shape)

def deriv_f1_x3(params, input_data):
    x7 = params[6, 0]
    x8 = params[7, 0]
    x10 = params[9, 0]
    x11 = params[10, 0]
    x13 = params[12, 0]
    xw, yw = input_data[:, 0], input_data[:, 1]
    return 1.0/(xw+x8 + yw*x11 + 1) * (x7*xw + yw*x10 + x13)

def deriv_f1_x4(params, input_data):
    x8 = params[7, 0]
    x11 = params[10, 0]
    xw, yw = input_data[:, 0], input_data[:, 1]
    return 1.0/(xw+x8 + yw*x11 + 1) * (x8*xw + yw*x11 + 1)

def deriv_f1_x5(params, input_data):
    xw, yw = input_data[:, 0], input_data[:, 1]
    return np.zeros(xw.shape)

def deriv_f1_x6(params, input_data):
    x1 = params[0, 0]
    x8 = params[7, 0]
    x11 = params[10, 0]
    xw, yw = input_data[:, 0], input_data[:, 1]
    return 1.0/(xw+x8 + yw*x11 + 1) * (x1*xw)

def deriv_f1_x7(params, input_data):
    x3 = params[2, 0]
    x8 = params[7, 0]
    x11 = params[10, 0]
    xw, yw = input_data[:, 0], input_data[:, 1]
    return 1.0/(xw+x8 + yw*x11 + 1) * (x3*xw)

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
    xw, yw = input_data[:, 0], input_data[:, 1]
    return 1.0/((xw+x8 + yw*x11 + 1)**2) * (x4*xw*(xw+x8 + yw*x11 + 1) - xw*(xw*(x1*x6+x3*x7+x4*x8) + yw*(x1*x9+x3*x10+x4*x11) + x1*x12+x3*x13+x4))

def deriv_f1_x9(params, input_data):
    x1 = params[0, 0]
    x8 = params[7, 0]
    x11 = params[10, 0]
    xw, yw = input_data[:, 0], input_data[:, 1]
    return 1.0/(xw+x8 + yw*x11 + 1) * yw*x1

def deriv_f1_x10(params, input_data):
    x3 = params[2, 0]
    x8 = params[7, 0]
    x11 = params[10, 0]
    xw, yw = input_data[:, 0], input_data[:, 1]
    return 1.0/(xw+x8 + yw*x11 + 1) * yw*x3

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
    xw, yw = input_data[:, 0], input_data[:, 1]
    return 1.0/((xw+x8 + yw*x11 + 1)**2) * (x11*yw*(xw+x8 + yw*x11 + 1) - yw*(xw*(x1*x6+x3*x7+x4*x8) + yw*(x1*x9+x3*x10+x4*x11) + x1*x12+x3*x13+x4))

def deriv_f1_x12(params, input_data):
    x1 = params[0, 0]
    x8 = params[7, 0]
    x11 = params[10, 0]
    xw, yw = input_data[:, 0], input_data[:, 1]
    return 1.0/(xw+x8 + yw*x11 + 1) * x1

def deriv_f1_x13(params, input_data):
    x3 = params[2, 0]
    x8 = params[7, 0]
    x11 = params[10, 0]
    xw, yw = input_data[:, 0], input_data[:, 1]
    return 1.0/(xw+x8 + yw*x11 + 1) * x3

def deriv_f2_x1(params, input_data):
    xw, yw = input_data[:, 0], input_data[:, 1]
    return np.zeros(xw.shape)

def deriv_f2_x2(params, input_data):
    x7 = params[6, 0]
    x10 = params[9, 0]
    x13 = params[12, 0]
    x8 = params[7, 0]
    x11 = params[10, 0]
    xw, yw = input_data[:, 0], input_data[:, 1]
    return 1.0/(xw+x8 + yw*x11 + 1) * (xw*x7+yw*x10+x13)

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
    xw, yw = input_data[:, 0], input_data[:, 1]
    return 1.0/(xw+x8 + yw*x11 + 1) * xw*x2

def deriv_f2_x8(params, input_data):
    x2 = params[1, 0]
    x5 = params[4, 0]
    x7 = params[6, 0]
    x8 = params[7, 0]
    x10 = params[9, 0]
    x11 = params[10, 0]
    x13 = params[12, 0]
    xw, yw = input_data[:, 0], input_data[:, 1]
    return 1.0/((xw+x8 + yw*x11 + 1)**2) * (x5*xw*(xw+x8 + yw*x11 + 1) - xw*(xw*(x2*x7+x5*x8) + yw*(x2*x10+x5*x11) + x2*x13+x5))

def deriv_f2_x9(params, input_data):
    xw, yw = input_data[:, 0], input_data[:, 1]
    return np.zeros(xw.shape)

def deriv_f2_x10(params, input_data):
    x2 = params[1, 0]
    x8 = params[7, 0]
    x11 = params[10, 0]
    xw, yw = input_data[:, 0], input_data[:, 1]
    return 1.0/(xw+x8 + yw*x11 + 1) * yw*x2

def deriv_f2_x11(params, input_data):
    x2 = params[1, 0]
    x5 = params[4, 0]
    x7 = params[6, 0]
    x8 = params[7, 0]
    x10 = params[9, 0]
    x11 = params[10, 0]
    x13 = params[12, 0]
    xw, yw = input_data[:, 0], input_data[:, 1]
    return 1.0/((xw+x8 + yw*x11 + 1)**2) * (x5*yw*(xw+x8 + yw*x11 + 1) - yw*(xw*(x2*x7+x5*x8) + yw*(x2*x10+x5*x11) + x2*x13+x5))

def deriv_f2_x12(params, input_data):
    xw, yw = input_data[:, 0], input_data[:, 1]
    return np.zeros(xw.shape)

def deriv_f2_x13(params, input_data):
    x2 = params[1, 0]
    x8 = params[7, 0]
    x11 = params[10, 0]
    xw, yw = input_data[:, 0], input_data[:, 1]
    return 1.0/(xw+x8 + yw*x11 + 1) * x2

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

    fun_list.append(f1_fun_list)
    fun_list.append(f2_fun_list)
    return fun_list

def deriv_function(params, input_data, param_index, fun_index):
    all_deriv_fun = myderiv_fun()

    params1 = params.copy()
    data_est_output = all_deriv_fun[fun_index][param_index](params1, input_data)
    return data_est_output



# calculating jacobian matrix,whose shape is (num_data,num_params)
def cal_Jacobian(params, input_data):
    num_params = np.shape(params)[0]
    num_data = np.shape(input_data)[0]
    J = np.zeros((num_data*2, num_params))
    for j in range(2):
        for i in range(num_params):
            #J[:, i] = list(cal_deriv(params, input_data, i))
            J[num_data*j:num_data*j+num_data, i] = list(deriv_function(params, input_data, i, j))
    return J


# calculating residual, whose shape is (num_data,1)
def cal_residual(params, input_data, output_data):
    data_est_output = my_Func(params, input_data)
    residual = np.linalg.norm(output_data - data_est_output, axis=1)
    return np.concatenate((residual, residual), axis=0).transpose()


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
    params = np.zeros((13, 1))
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
    num_data = 200  # set the data number
    data_input, data_output = generate_data(params, num_data)  # generate data as requested

    # set the init params for LM algorithm
    params[0, 0] = 6.0
    params[1, 0] = 0.3
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

    # using LM algorithm estimate params
    num_iter = 100  # the number of iteration
    est_params = LM(num_iter, params, data_input, data_output)
    print(est_params)
    a_est = est_params[0, 0]
    b_est = est_params[1, 0]

    # 老子画个图看看状况
    plt.scatter(data_input, data_output, color='b')
    x = np.arange(0, 100) * 0.1  # 生成0-10的共100个数据，然后设置间距为0.1
    plt.plot(x, a_est * np.exp(b_est * x), 'r', lw=1.0)
    plt.xlabel("2018.06.13")
    plt.savefig("result_LM.png")
    plt.show()

    plt.plot(residual_memory)
    plt.xlabel("2018.06.13")
    plt.savefig("error-iter.png")
    plt.show()


if __name__ == '__main__':
    main()
