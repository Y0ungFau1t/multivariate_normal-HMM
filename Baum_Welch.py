import numpy as np
from scipy.stats import multivariate_normal
from sklearn import mixture
from HMM import emission, Viterbi
import time

def scaled_forward(O, T, u, cov, pi, S = [0,1,2]):
    O = np.array(O)
    T = np.array(T)
    alpha = np.zeros((O.shape[0], T.shape[0]))
    alpha_hat = np.zeros((O.shape[0], T.shape[0]))
    G = np.zeros((O.shape[0]))
    #start = time.time()
    b = emission(O,u,cov,S)
    for j in range(T.shape[0]):
        #start1 = time.time()
        #alpha[0, j] = pi[j] * emission(j, O[0], u, cov, S) #alpha[0,i] = pi[i]*b_i(Y_0)
        alpha[0, j] = pi[j] * b[0,j]
        #end1 = time.time()
        #print("emission calculate time : ", end1 - start1)

    G[0] = np.sum(alpha[0,:]) #G[0] = sum_(i=1)^(3) alpha[0,i]
    alpha_hat[0, :] = alpha[0,:]/G[0] #alpha_hat[0,i] = pi[i]*b_i(Y_0)/G[0]
    for t in range(1, O.shape[0]):
        #start2 = time.time()
        for j in range(T.shape[0]):
            #alpha[t, j] = alpha_hat[t - 1,:] @ T[:, j] * emission(j, O[t], u, cov, S) #alpha[t,j] = b_j(Y_t)*sum_(i=1)^(3){alpha_hat[t-1,i]*a_i_j
            alpha[t, j] = alpha_hat[t - 1, :] @ T[:, j] * b[t,j]
        G[t] = sum(alpha[t,:]) #G[t] = sum_(j=1)^(3) alpha[t,j]
        alpha_hat[t, :] = alpha[t,:]/G[t] #alpha_hat[t,j] = alpha[t,j]/G[t]
        #end2 = time.time()
        #print('emission calculate time :', end2 - start2)
    #end = time.time()
    #print("in total: ", end - start)
    return alpha_hat, G

def scaled_backward(O, T, u, cov, S = [0,1,2]):
    O = np.array(O)
    T = np.array(T)
    beta = np.zeros((O.shape[0], T.shape[0]))
    beta_hat = np.zeros((O.shape[0], T.shape[0]))
    # setting beta[T,i] = 1 and beta_hat[T,i] = 1, i = 0,1,2 here
    beta[O.shape[0] - 1] = np.ones((T.shape[0]))
    beta_hat[O.shape[0] - 1] = np.ones((T.shape[0]))
    G = np.zeros((O.shape[0]))
    b = emission(O, u, cov, S)
    # Loop in backward way from T-1 to
    # Due to python indexing the actual loop will be T-2 to 0
    for t in range(O.shape[0] - 2, -1, -1):
        for j in range(T.shape[0]):
            list = []
            for i in range(T.shape[0]):
                #list.append(beta_hat[t+1][i] * emission(i, O[t+1], u, cov, S)) #[beta_hat[t+1,i]*b_i(Y_t+1)]_(i in {0,1,2})
                list.append(beta_hat[t + 1][i] * b[t+1,i])
                #print(emission(i, O[t+1], u, cov, S))
            list = np.array(list)
            #print(list)
            beta[t, j] = list @ T[j, :] #beta[t,j] = sum_(i = 1)^(3) a_ji*beta_hat[t+1,i]*b_i(Y_t+1)
        G[t] = sum(beta[t,:]) #G[t] = sum_(i=1)^(3) beta[t,j]
        beta_hat[t,:] = beta[t,:]/G[t] #beta_hat[t,j] = beta[t,j]/G[t]
    return beta_hat

def scaled_forward_b(O, T, b, pi):
    O = np.array(O)
    T = np.array(T)
    alpha = np.zeros((O.shape[0], T.shape[0]))
    alpha_hat = np.zeros((O.shape[0], T.shape[0]))
    G = np.zeros((O.shape[0]))
    #start = time.time()
    for j in range(T.shape[0]):
        #start1 = time.time()
        #alpha[0, j] = pi[j] * emission(j, O[0], u, cov, S) #alpha[0,i] = pi[i]*b_i(Y_0)
        alpha[0, j] = pi[j] * b[0,j]
        #end1 = time.time()
        #print("emission calculate time : ", end1 - start1)

    G[0] = np.sum(alpha[0,:]) #G[0] = sum_(i=1)^(3) alpha[0,i]
    alpha_hat[0, :] = alpha[0,:]/G[0] #alpha_hat[0,i] = pi[i]*b_i(Y_0)/G[0]
    for t in range(1, O.shape[0]):
        #start2 = time.time()
        for j in range(T.shape[0]):
            #alpha[t, j] = alpha_hat[t - 1,:] @ T[:, j] * emission(j, O[t], u, cov, S) #alpha[t,j] = b_j(Y_t)*sum_(i=1)^(3){alpha_hat[t-1,i]*a_i_j
            alpha[t, j] = alpha_hat[t - 1, :] @ T[:, j] * b[t,j]
        G[t] = sum(alpha[t,:]) #G[t] = sum_(j=1)^(3) alpha[t,j]
        alpha_hat[t, :] = alpha[t,:]/G[t] #alpha_hat[t,j] = alpha[t,j]/G[t]
        #end2 = time.time()
        #print('emission calculate time :', end2 - start2)
    #end = time.time()
    #print("in total: ", end - start)
    return alpha_hat, G

def scaled_backward_b(O, T, b):
    O = np.array(O)
    T = np.array(T)
    beta = np.zeros((O.shape[0], T.shape[0]))
    beta_hat = np.zeros((O.shape[0], T.shape[0]))
    # setting beta[T,i] = 1 and beta_hat[T,i] = 1, i = 0,1,2 here
    beta[O.shape[0] - 1] = np.ones((T.shape[0]))
    beta_hat[O.shape[0] - 1] = np.ones((T.shape[0]))
    G = np.zeros((O.shape[0]))
    # Loop in backward way from T-1 to
    # Due to python indexing the actual loop will be T-2 to 0
    for t in range(O.shape[0] - 2, -1, -1):
        for j in range(T.shape[0]):
            list = []
            for i in range(T.shape[0]):
                #list.append(beta_hat[t+1][i] * emission(i, O[t+1], u, cov, S)) #[beta_hat[t+1,i]*b_i(Y_t+1)]_(i in {0,1,2})
                list.append(beta_hat[t + 1][i] * b[t+1,i])
                #print(emission(i, O[t+1], u, cov, S))
            list = np.array(list)
            #print(list)
            beta[t, j] = list @ T[j, :] #beta[t,j] = sum_(i = 1)^(3) a_ji*beta_hat[t+1,i]*b_i(Y_t+1)
        G[t] = sum(beta[t,:]) #G[t] = sum_(i=1)^(3) beta[t,j]
        beta_hat[t,:] = beta[t,:]/G[t] #beta_hat[t,j] = beta[t,j]/G[t]
    return beta_hat


def scaled_baum_welch(O, trans, u, cov, pi, S = [0,1,2], n_iter=100):
    O = np.array(O)
    trans = np.array(trans)
    M = trans.shape[0]
    T = len(O)
    likelihood_hist = -np.inf

    for n in range(n_iter):
        ###estimation step
        b = emission(O, u, cov, S)
        start = time.time()
        #alpha, G = scaled_forward(O, trans, u, cov, pi, S)
        alpha, G = scaled_forward_b(O, trans, b, pi)
        end = time.time()
        print("forward finish: ", end - start)
        #print(G)
        start = time.time()
        #beta = scaled_backward(O, trans, u, cov, S)
        beta = scaled_backward_b(O, trans, b)
        end = time.time()
        print("backward finish: ", end - start)
        xi = np.zeros((M, M, T - 1))
        #P_SS_Y = np.zeros((M, M, T - 1))
        for t in range(T - 1):
            # joint probab of observed data up to time t @ transition prob * emisssion prob as t+1 @
            # joint probab of observed data from time t+1
            denominator = (alpha[t, :].T @ trans * b[t + 1, :]) @ beta[t + 1, :]
            for i in range(M):
                numerator = alpha[t, i] * trans[i, :] * b[t + 1, :] * beta[t + 1, :].T
                xi[i, :, t] = numerator / denominator #Xi_ij(t) = P(S_t = i,S_(t+1) = j|Y)
                #P_SS_Y[i, :, t] = numerator #P(S_t = i, S_(t+1) = j, Y)
        end = time.time()
        print('xi is calculated: ', end - start)
        start = time.time()
        gamma = np.sum(xi, axis=1) #Gamma_i(t) = P(S_t = i|Y)
        #P_S_Y = np.sum(P_SS_Y, axis=1) #P(S_t = i, Y)
        end = time.time()
        print('gamma is calculated: ', end - start)
        ### maximization step
        start = time.time()
        trans = np.sum(xi, 2) / np.sum(gamma, axis=1).reshape((-1, 1))
        end = time.time()
        print('trans is renewed: ', end - start)
        gamma = np.hstack((gamma, np.sum(xi[:, :, T - 2], axis=0).reshape((-1, 1)))) # Add additional T'th element in gamma
        #P_S_Y = np.hstack((P_S_Y, np.sum(P_SS_Y[:, :, T - 2], axis=0).reshape((-1, 1))))
        start = time.time()
        best_path = np.argmax(gamma, axis=0)
        end = time.time()
        print('best path is given: ', end - start)
        #best_path = Viterbi(O, S, pi, trans, u, cov)
        log_likelihood = np.sum(np.log(G))

        if log_likelihood < likelihood_hist:
            print("log_likelihood decreased")
            break

        if log_likelihood - likelihood_hist < 0.000001:
            print(n," steps' log likelihood:",log_likelihood, "converged")
            break

        #print(n," steps' log likelihood:",log_likelihood)
        likelihood_hist = log_likelihood
        print(likelihood_hist)
        start = time.time()
        pi = gamma[:, 0]
        for i in range(M):
            u[i] = np.dot(gamma[i], O) / np.sum(gamma[i]) #u[i] = sum_(t = 1)^(T){Y_t*P(S_t = i|Y)}/sum_(t=1)^(T)P(S_t = i|Y)
        for i in range(M):
            try:
                sigma = np.zeros((len(O[0]), len(O[0])))
            except:
                sigma = 0
            for j in range(T):
                sigma += np.outer((O[j] - u[i]), (O[j] - u[i])) * gamma[i][j]
            cov[i] = sigma / np.sum(gamma[i])
        end = time.time()
        print('pi&u&cov is renewed: ', end - start)

    return trans, u, cov, pi, best_path, log_likelihood
