import numpy
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_digits
import matplotlib.colors as mcolors
from scipy.optimize import minimize


#################################################
colors = []
for key, value in mcolors.TABLEAU_COLORS.items():
    colors.append(value)
    
#################################
#general functions
def splitData(X, y):
    t = 5
    n = [i * int(len(X)/t) for i in range(t + 1)]
    return [[X[n[i] : n[i + 1]], y[n[i] : n[i + 1]]]  for i in range(t)]

def quantile(v, alpha):
    m = int(numpy.ceil((1 - alpha) * (len(v) + 1))) - 1
    v = numpy.sort(v, axis = 0)
    return v[m]

def H(p, eps=.0001):
    return -sum([s * numpy.log(s + eps) for s in p])

def sigma(x):
    return 1/(1 + numpy.exp(-x))

def sigmaPrime(x):
    return sigma(x) * (1 - sigma(x))

def F1score(sets, y):
    TP, FP, TN, FN = 0, 0, 0, 0
    mclasses = max(y) + 1
    for i in range(len(sets)):
        interval = sets[i]
        positives = [j for j in range(mclasses) if j in interval]
        negatives = [j for j in range(mclasses) if j not in interval]
        TP = TP + sum([1 for j in positives if j == y[i]])
        FP = FP + sum([1 for j in positives if j != y[i]])
        TN = TN + sum([1 for j in negatives if j != y[i]])  
        FN = FN + sum([1 for j in negatives if j == y[i]]) 
    F1 = 2 * TP/(2 * TP + FP + FN)
    return F1

def evaluateCPclass(d1, d2, theta, alpha):
    x, a, g, allas, allg, y, h = d1 
    B = [[b(allas[i][m], allg[i][m], h[i], theta) for m in range(max(y) + 1)] for i in range(len(allas))]
    q = quantile([B[i][y[i]] for i in range(len(y))], alpha)
    
    x, a, g, allas, allg, y, h = d2
    B = [[b(allas[i][m], allg[i][m], h[i], theta) for m in range(max(y) + 1)] for i in range(len(allas))]
    sets = [[m for m in range(max(y) + 1) if B[i][m] < q] for i in range(len(y))]
    F1 = F1score(sets, y)
    sizes = numpy.sum([len(sets[i]) for i in range(len(sets))])/len(y)
    val = numpy.sum([1 for i in range(len(y)) if y[i] in sets[i]])/len(y)
    return sizes, F1, val

#######################################
#experiment functions
def classifier(D):
    X, Y = D
    rf = RandomForestClassifier(max_depth=20, random_state=0)
    rf.fit(X, Y.squeeze())
    return rf    

def approximator(D):
    X, A, Y = D
    print(X.shape, A.shape, Y.shape)
    rf = []
    for m in range(max(Y) + 1): 
        rf.append(RandomForestRegressor(max_depth=25, random_state=0))
        x = numpy.array([X[i] for i in range(len(Y)) if Y[i] == m])
        a = numpy.array([A[i] for i in range(len(Y)) if Y[i] == m])
        rf[-1].fit(x, a.squeeze())
    return rf    

def prepareData(dataset, rfClass, rg):
    x = dataset[0]
    y = dataset[1]
    allprobs = rfClass.predict_proba(x)
    allas = 1 - allprobs
    a = numpy.array([1 - allprobs[i, y[i]].squeeze() for i in range(len(allprobs))])
    allg = numpy.array([rg[i].predict(x) for i in range(max(y) + 1)]).transpose()
    g = numpy.array([allg[i, y[i]].squeeze() for i in range(len(allprobs))])
    h = numpy.array([H(1 - allg[i, :]).squeeze() for i in range(len(allg))])
    return x, a, g, allas, allg, y, h

def r(t, g, h):
    r1 = t[0] + t[1] * numpy.power(abs(g), t[2]) + t[3] * numpy.power(abs(h), t[4])
    r2 = t[5] + t[6] * numpy.power(abs(g), t[7]) + t[8] * numpy.power(abs(h), t[9])
    return r1, r2
    
def b(a, g, h, theta):
    r1, r2 = r(theta, g, h)
    return a * numpy.exp(-r1) - r2 

def ML(theta, d1, d2, rho = 0):
    x1, a1, g1, allas1, allg1, y1, h1 = d1
    x2, a2, g2, allas2, allg2, y2, h2 = d2
    mclass = max(y1) + 1
    K = theta[-1]
    B1 = numpy.array([b(allas1[i1, y1[i1]], allg1[i1, y1[i1]], h1[i1], theta) for i1 in range(len(y1))]).reshape([len(y1), 1])
    B2 = numpy.array([b(allas2[i2, y2[i2]], allg2[i2, y2[i2]], h2[i2], theta) for i2 in range(len(y2))]).reshape([len(y2), 1])
    s1 = - numpy.log(K * K/len(B1) + 1e-4)/2
    M = B1 - B2.transpose()
    s2 = K * K * numpy.sum(M * M)
    return s1 + s2  + rho * theta @ theta


def smoothSize(theta, d1, d2, rho = 0):
    x1, a1, g1, allas1, allg1, y1, h1 = d1
    x2, a2, g2, allas2, allg2, y2, h2 = d2
    mclass = max(y1) + 1
    K = theta[-1]
    M = .1
    B1 = numpy.array([[b(allas1[i2, k], allg1[i2, k], h1[i2], theta) for k in range(mclass)] for i2 in range(len(y2))])
    B2 = numpy.array([b(allas2[i2, y2[i2]], allg2[i2, y2[i2]], h2[i2], theta) for i2 in range(len(y2))])
    b2 = [x/sum(B2) for x in B2]
    w = numpy.exp([K * K * s for s in b2]) + 1e-4
    w = numpy.diag(w / sum(w))
    arrays = [B2 for i in range(10)]
    B2M = numpy.stack(arrays, axis=1)
    S = sigma(M * (B2M - B1))
    ell = numpy.sum(w @ S)
    return ell + rho * theta @ theta

def direct(theta, d1, d2, alpha = .1):
    x1, a1, g1, allas1, allg1, y1, h1 = d1
    x2, a2, g2, allas2, allg2, y2, h2 = d2

    B1 = numpy.array([b(allas1[i1, y1[i1]], allg1[i1, y1[i1]], h1[i1], theta) for i1 in range(len(y1))]).reshape([len(y1), 1])
    q = quantile([B1[i] for i in range(len(y1))], alpha)
    
    B2 = [[b(allas2[i2][m], allg2[i2][m], h2[i2], theta) for m in range(max(y2) + 1)] for i2 in range(len(allas2))]
    sets = [[m for m in range(max(y2) + 1) if B2[i2][m] < q] for i2 in range(len(y2))]
    sizes = numpy.sum([len(sets[i]) for i in range(len(sets))])/len(y2)
    return sizes 

def evaluateCPclass(d1, d2, theta, alpha):
    x, a, g, allas, allg, y, h = d1 
    B = [[b(allas[i][m], allg[i][m], h[i], theta) for m in range(max(y) + 1)] for i in range(len(allas))]
    q = quantile([B[i][y[i]] for i in range(len(y))], alpha)
    
    x, a, g, allas, allg, y, h = d2
    B = [[b(allas[i][m], allg[i][m], h[i], theta) for m in range(max(y) + 1)] for i in range(len(allas))]
    sets = [[m for m in range(max(y) + 1) if B[i][m] < q] for i in range(len(y))]
    F1 = F1score(sets, y)
    sizes = numpy.sum([len(sets[i]) for i in range(len(sets))])/len(y)
    val = numpy.sum([1 for i in range(len(y)) if y[i] in sets[i]])/len(y)
    return sizes, F1, val
