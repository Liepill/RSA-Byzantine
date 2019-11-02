import numpy as np
import random
import matplotlib.pyplot as plt
import time
np.set_printoptions(threshold='nan')

num_class = 10  # number of classes
num_feature = 28 * 28  # number of features
num_train = 60000  # number of train samples
num_test = 10000  # number of test samples
num_machines = 20  # number of workers
batch_size = 32

num_iter = 5000  # number of iterations
exit_byzantine = True
num_byz = 8  # number of Byzantine workers


def cal_total_grad(X, Y, theta, weight_lambda):

    """
    :param X: shape(num_samples, features + 1)
    :param Y: labels' one_hot array, shape(num_samples, num_classes)
    :param theta: shape (num_classes, feature+1)
    :param weight_lambda: scalar
    :return: grad, shape(num_classes, feature+1)
    """
    m = X.shape[0]
    t = np.dot(theta, X.T) #(num_classes, num_samples)
    t = t - np.max(t, axis=0)
    pro = np.exp(t) / np.sum(np.exp(t), axis=0)
    total_grad = -np.dot((Y.T - pro), X) / m #+ weight_lambda * theta
    return total_grad


def cal_loss(X, Y, theta, weight_lambda):

    m = X.shape[0]
    t1 = np.dot(theta, X.T)
    t1 = t1 - np.max(t1, axis=0)
    t = np.exp(t1)
    tmp = t / np.sum(t, axis=0)
    loss = -np.sum(Y.T * np.log(tmp)) / m + weight_lambda * np.sum(theta ** 2) / 2
    return loss


def cal_acc(test_x, test_y, theta):

    num = 0
    m = test_x.shape[0]
    for i in range(m):
        t1 = np.dot(theta, test_x[i])
        t1 = t1 - np.max(t1, axis=0)
        pro = np.exp(t1) / np.sum(np.exp(t1), axis=0)
        index = np.argmax(pro)
        if index == test_y[i]:
            num += 1
    acc = float(num) / m
    return acc


def cal_max_norm_grad(theta):
    if np.all(theta == 0):
        return theta
    tmp = np.abs(theta)
    re = np.where(tmp == np.max(tmp))
    row = re[0][0]
    col = re[1][0]
    max_val = tmp[row, col]
    tmp_theta = np.zeros_like(theta)
    n = len(re[0])
    theta[tmp != np.max(tmp)] = 0
    theta[theta == -max_val] = -1.0 / n
    theta[theta == max_val] = 1.0 / n
    return theta


def cal_var(theta):
    mean_theta = np.mean(theta, axis=0)
    mean_arr = np.tile(mean_theta, (theta.shape[0], 1))
    tmp = theta - mean_arr
    var = np.trace(np.dot(tmp, tmp.T))
    return var


def huber_loss_grad(e, d):

    t = (np.abs(e) <= d) * e
    e[np.abs(e) <= d] = 0
    grad = t + d * np.sign(e)
    return grad


class Machine:
    def __init__(self, data_x, data_y, machine_id):
        """Initializes the machine with the data
        Accepts data, a numpy array of shape :(num_samples/num_machines, dimension)
        data_x : a numpy array has shape :num_samples/num_machines, dimension)
        data_y: a list of length 'num_samples/num_machine', the label of the data_x"""

        self.data_x = data_x
        self.data_y = data_y
        self.machine_id = machine_id

    def update(self, theta0, theta, alpha, l1_lambda, weight_lambda, delta):
        """Calculates gradient with a randomly selected sample, given the current theta
         Accepts theta, a np array with shape of (dimension,)
         Returns the calculated gradient"""
        m = self.data_x.shape[0]
        # print "machine%d:"%(self.machine_id), m
        id = random.randint(0, m - batch_size)
        grad_f = cal_total_grad(self.data_x[id:(id + batch_size)], self.data_y[id:(id + batch_size)], theta,
                                weight_lambda)
        # L1 norm
        grad = grad_f / num_machines + l1_lambda * np.sign(theta - theta0)

        # # # L2 norm
        # tmp = np.linalg.norm(theta - theta0)
        # if tmp == 0:
        #     tmp = 1e-5
        # grad = grad_f / num_machines + l1_lambda * (theta - theta0) / tmp

        # max_norm
        # grad = grad_f / num_machines + l1_lambda * cal_max_norm_grad(theta - theta0)
        #
        #huber loss grad
        # grad = grad_f / num_machines + l1_lambda * huber_loss_grad(theta - theta0, delta)

        new_theta = theta - alpha * grad

        return new_theta

class Parameter_server:
    def __init__(self):
        """Initializes all machines"""
        self.theta0_li = []
        self.theta_li = [] #list that stores each theta, grows by one iteration
        self.acc_li = []
        self.grad_li = []
        self.grad_norm = []
        self.theta0_star_norm = []
        self.acc_li = []
        self.loss_li = []
        self.theta_li_diff = []
        self.theta0_li_diff = []
        self.time_li = []
        self.var_li = []

        train_img = np.load('../data/mnist/train_img.npy')  # shape(60000, 784)
        train_lbl = np.load('../data/mnist/train_lbl.npy')  # shape(60000,)
        one_train_lbl = np.load('../data/mnist/one_train_lbl.npy')  # shape(60000, 10)
        test_img = np.load('../data/mnist/test_img.npy')  # shape(10000, 784)
        test_lbl = np.load('../data/mnist/test_lbl.npy')  # shape(10000,)

        bias_train = np.ones(num_train)
        train_img_bias = np.column_stack((train_img, bias_train))

        bias_test = np.ones(num_test)
        test_img_bias = np.column_stack((test_img, bias_test))

        self.test_img_bias = test_img_bias
        self.test_lbl = test_lbl
        self.train_img_bias = train_img_bias
        self.one_train_lbl = one_train_lbl
        self.train_lbl = train_lbl

        samples_per_machine = num_train / num_machines
        self.machines = []
        #########  i.i.d case
        for i in range(num_machines):
            new_machine = Machine(train_img_bias[i * samples_per_machine:(i + 1) * samples_per_machine, :],
                                  one_train_lbl[i * samples_per_machine:(i + 1) * samples_per_machine, :], i)
            self.machines.append(new_machine)


    def broadcast(self, theta0, theta_li, alpha, l1_lambda, weight_lambda, delta):
        """Broadcast theta
        Accepts theta, a numpy array of shape:(dimension,)
        Return a list of length 'num_machines' containing the updated theta of each machine"""

        new_theta_li = []
        for i, mac in enumerate(self.machines):
            new_theta_li.append(mac.update(theta0, theta_li[i], alpha, l1_lambda, weight_lambda, delta))
        tmp = np.zeros_like(theta0)
        for i in range(len(theta_li)):
            # L1 norm
            if exit_byzantine == False:
                tmp += np.sign(theta0 - new_theta_li[i])
            else:
                if i < num_machines - num_byz:
                    tmp += np.sign(theta0 - new_theta_li[i])
                else:
                    gaussian_attack = np.random.standard_normal((num_class, num_feature+1))*10000
                    tmp += np.sign(theta0 - gaussian_attack)

            # # # L2 norm
            # if exit_byzantine == False:
            #     frac_tmp = np.linalg.norm(theta0 - new_theta_li[i])
            #     if frac_tmp == 0:
            #         frac_tmp = 1e-5
            #     tmp += (theta0 - new_theta_li[i]) / frac_tmp
            # else:
            #     if i < num_machines - num_byz:
            #         frac_tmp = np.linalg.norm(theta0 - new_theta_li[i])
            #         if frac_tmp == 0:
            #             frac_tmp = 1e-5
            #         tmp += (theta0 - new_theta_li[i]) / frac_tmp
            #     else:
            #         gaussian_attack = np.random.standard_normal((num_class, num_feature + 1)) * 10000
            #         frac_tmp = np.linalg.norm(theta0 - gaussian_attack)
            #         if frac_tmp == 0:
            #             frac_tmp = 1e-5
            #         tmp += (theta0 - gaussian_attack) / frac_tmp


            # max norm
            # if exit_byzantine == False:
            #     tmp += cal_max_norm_grad(theta0 - new_theta_li[i])
            # else:
            #     if i < num_machines - num_byz:
            #         tmp += cal_max_norm_grad(theta0 - new_theta_li[i])
            #     else:
            #         gaussian_attack = np.random.standard_normal((num_class, num_feature + 1)) * 10000
            #         tmp += cal_max_norm_grad(theta0 - gaussian_attack)
            #huber loss grad
            # tmp += huber_loss_grad(theta0 - new_theta_li[i], delta)

        new_theta0 = theta0 - alpha * (l1_lambda * tmp + weight_lambda * theta0)
        return new_theta0, new_theta_li

    def train(self, init_theta0, init_theta, alpha, l1_lambda, weight_lambda):
        """Peforms num_iter rounds of update, appends each new theta to theta_li
        Accepts the initialed theta, a numpy array has shape:(dimension,)"""

        self.theta0_li.append(init_theta0)
        self.theta_li.append(init_theta)
        delta = 0.1
        d = 0.001
        for i in range(num_iter):
            alpha = d / np.sqrt(i + 1)
            rec_theta0, rec_theta = self.broadcast(self.theta0_li[-1], self.theta_li[-1], alpha, l1_lambda, weight_lambda, delta)
            if (i + 1) % 1000 == 0:
                self.theta_li = []
                self.theta0_li = []
            self.theta0_li.append(rec_theta0)
            self.theta_li.append(rec_theta)

            # loss = cal_loss(self.train_img_bias, self.one_train_lbl, rec_theta0, weight_lambda)
            # total_grad = cal_total_grad(self.train_img_bias, self.one_train_lbl, rec_theta0, weight_lambda) + weight_lambda * rec_theta0
            # # total_grad = cal_total_grad(self.train_img_bias, self.one_train_lbl, rec_theta[0][0],weight_lambda) + weight_lambda * rec_theta[0][0]
            # self.grad_norm.append(np.linalg.norm(total_grad))
            # self.loss_li.append(loss)
            
            if (i + 1) % 10 == 0:
                acc = cal_acc(self.test_img_bias, self.test_lbl, rec_theta0)
                self.acc_li.append(acc)
                print "step:", i, " acc:", acc

            # compute variation
            theta_tmp = []
            for k in range(num_machines - num_byz):
                theta_tmp.append(rec_theta[k])
            theta_tmp = np.array(theta_tmp)
            theta_tmp = theta_tmp.reshape(num_machines - num_byz, 10*785)
            rec_theta0 = rec_theta0.reshape(1, 10*785)
            var_theta = np.row_stack((theta_tmp, rec_theta0))
            self.var_li.append(cal_var(var_theta))

        print("train end!")

    def plot_curve(self):
        """plot the loss curve and the acc curve
        save the learned theta to a numpy array and a txt file"""

        # s1 = 'L2/gaussian/q8'
        # np.save('./result/RSGD/fault/' + s1 + '/acc.npy', self.acc_li)
        # # np.save('./result/RSGD/no_fault/same_digit/' + s1 + '/grad_norm.npy', self.grad_norm)
        # np.save('./result/RSGD/fault/' + s1 + '/var_li.npy', self.var_li)

        plt.plot(np.arange(len(self.acc_li)) * 10, self.acc_li)
        plt.xlabel('iter')
        plt.ylabel('accuracy')
        # plt.title(s1)
        # plt.savefig('./result/RSGD/fault/' + s1 + '/acc.png')
        plt.show()

        # plt.semilogy(np.arange(num_iter), self.grad_norm)
        # plt.xlabel('iter')
        # plt.ylabel('log||grad||')
        # # plt.title(s1)
        # plt.savefig('./result/RSGD/no_fault/same_digit/' + s1 + '/grad_norm.png')
        # plt.show()

        plt.semilogy(np.arange(num_iter), self.var_li)
        plt.xlabel('iter')
        plt.ylabel('log||var||')
        # plt.savefig('./result/RSGD/fault/' + s1 + '/var.png')
        plt.show()


def init():
    server = Parameter_server()
    return server


def main():
    server = init()
    init_theta0 = np.zeros((num_class, num_feature + 1))
    init_theta = []
    for i in range(num_machines):
        init_theta.append(np.zeros((num_class, num_feature + 1)))
    alpha = 0.001
    l1_lambda = 0.07
    weight_lambda = 0.01
    server.train(init_theta0, init_theta, alpha, l1_lambda, weight_lambda)
    server.plot_curve()


if __name__ == "__main__":
    main()

