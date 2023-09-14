import numpy as np
from scipy.stats import multivariate_normal, norm
import time

def emission(O,u,cov,S = [0,1,2]):
    start = time.time()
    assert (len(S)==len(u) and len(S) == len(cov))
    output = np.zeros((len(O),len(S)))
    #print(output.shape)
    for t in range(len(O)):
        for j in range(len(S)):
            #output[t,j] = norm.pdf(O[t],loc = u[j],scale = np.sqrt(cov[j]))
            output[t, j] = multivariate_normal.pdf(O[t], mean=u[j], cov=cov[j])
    end = time.time()
    print('new emission time: ', end - start)
    return output

def Viterbi(O,S,pi,T,u,cov):
    assert (len(S) == len(u) and len(S) == len(cov))
    trellis = np.zeros((len(S),len(O)))
    pointers = np.zeros((len(S),len(O)))
    for s in range(len(S)):
        trellis[s,0] = pi[s]*emission(s,O[0],u,cov,S)
    for o in range(1,len(O)):
        for s in range(len(S)):
            #print([trellis[k,o-1] for k in range(len(S))])
            #print([T[k,s] for k in range(len(S))])
            #print(emission(s,O[o],u,cov,S))
            k = np.argmax([(trellis[k,o-1]*T[k,s]*emission(s,O[o],u,cov,S)) for k in range(len(S))])
            trellis[s,o] = trellis[k,o-1]*T[k,s]*emission(s,O[o],u,cov,S)
            pointers[s,o] = int(k)
    best_path = list()
    k = np.argmax([trellis[k,len(O)-1] for k in range(len(S))])
    k = int(k)
    for o in range(len(O)-1,-1,-1):
        #print(k,type(k))
        best_path.insert(0,S[k])
        k = pointers[k,o]
        k = int(k)
    return best_path
