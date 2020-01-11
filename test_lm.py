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
threshold_stop = 10 ** -20
threshold_step = 10 ** -20
threshold_residual = 10 ** -15
residual_memory = []


# construct a user function
def my_Func(params, input_data):
    x1 = params[0, 0]
    x3 = params[2, 0]
    x2 = params[1, 0]
    xw, yw = input_data[:, 0], input_data[:, 1]
    out_x =  x1 * xw + x2*yw
    out_y =  x1**2*xw + x3
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
    xw, yw = input_data[:, 0], input_data[:, 1]
    return xw

def deriv_f1_x2(params, input_data):
    x2 = params[1, 0]
    xw, yw = input_data[:, 0], input_data[:, 1]
    return yw

def deriv_f1_x3(params, input_data):
    xw, yw = input_data[:, 0], input_data[:, 1]
    return np.zeros(xw.shape)


def deriv_f2_x1(params, input_data):
    x1 = params[0, 0]
    xw, yw = input_data[:, 0], input_data[:, 1]
    return 2 * x1 *xw

def deriv_f2_x2(params, input_data):
    xw, yw = input_data[:, 0], input_data[:, 1]
    return np.zeros(xw.shape)



def deriv_f2_x3(params, input_data):
    xw, yw = input_data[:, 0], input_data[:, 1]
    return np.ones(xw.shape)


def myderiv_fun():
    f1_fun_list = []
    f2_fun_list = []
    fun_list = []
    f1_fun_list.append(deriv_f1_x1)
    f1_fun_list.append(deriv_f1_x2)
    f1_fun_list.append(deriv_f1_x3)


    f2_fun_list.append(deriv_f2_x1)
    f2_fun_list.append(deriv_f2_x2)
    f2_fun_list.append(deriv_f2_x3)

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
    J = np.zeros((num_data, num_params))
    residual_u_x, residual_v_y = cal_residual_u_x_v_y(params, input_data)
    for i in range(num_params):
        deriv_for_param_i = residual_u_x * deriv_function(params, input_data, i, 0) + residual_v_y * deriv_function(params, input_data, i, 1)
        J[:, i] = list(deriv_for_param_i)
    return J


# calculating residual, whose shape is (num_data,1)
def cal_residual(params, input_data, output_data):
    data_est_output = my_Func(params, input_data)
    residual = np.linalg.norm(output_data - data_est_output, axis=1).reshape(input_data.shape[0], 1)
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
            if (stop or rou > 0):
                break;

    return params


def main():
    # set the true params for generate_data() function
    params = np.zeros((3, 1))
    params[0, 0] = 5.0
    params[1, 0] = 2
    params[2, 0] = 4.0

    num_data = 200  # set the data number
    data_input, data_output = generate_data(params, num_data)  # generate data as requested

    # set the init params for LM algorithm
    params[0, 0] = 0.0
    params[1, 0] = 2.0
    params[2, 0] = 0.0


    # using LM algorithm estimate params
    num_iter = 100  # the number of iteration
    est_params = LM(num_iter, params, data_input, data_output)
    print(est_params)
    a_est = est_params[0, 0]
    b_est = est_params[1, 0]
    c_est = est_params[2, 0]
    print(a_est, b_est, c_est)
    # 老子画个图看看状况
    estimation = my_Func(est_params, data_input)
    plt.scatter(estimation[:, 0], estimation[:, 1], color='b')
    plt.scatter(data_output[:, 0], data_output[:, 1], color='r')
    x = np.arange(0, 100) * 0.1  # 生成0-10的共100个数据，然后设置间距为0.1
    #plt.plot(x, a_est * np.exp(b_est * x), 'r', lw=1.0)
    plt.xlabel("2018.06.13")
    plt.savefig("result_LM.png")
    plt.show()

    plt.plot(residual_memory)
    plt.xlabel("2018.06.13")
    plt.savefig("error-iter.png")
    plt.show()


if __name__ == '__main__':
    main()
