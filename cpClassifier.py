import numpy
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_digits
import matplotlib.colors as mcolors
from scipy.optimize import minimize
from wscClass import *



class transformScores:
    def __init__(self, classifier, model, X, y, rho = 0):
        if model[-1] == '+':
            self.more = 1
        else:
            self.more = 0
        self.classifier = classifier
        self.n, self.K = self.classifier.predict_proba(X).shape
        self.rg = self.approximator(X, y)
        self.npars = 2 
        if self.more: self.npars = 2 * (self.K + 1)
        self.rho = rho# * self.npars
        self.model = model
        if model[:2] == 'ER':
            self.b = self.bER
        if model[:2] == 'HR':
            self.b = self.bHR
        if model[:4] == 'temp':
            self.b = self.bTemp
        if model[:3] == 'off':
            self.b = self.bOff
        if model == 'unweighted':
            self.b = self.bBase
            self.theta = numpy.zeros(self.npars) 
            print(self.model,  ': done')
            print('status, nit, theta: ', None, None, self.theta)
        if model != 'unweighted': 
            self.theta = self.fitTransformation(X, y)

    def basicScore(self, p, eps = 1e-2):
        return 1 - p 

    def approximator(self, X, y):
        A = self.basicScore(self.classifier.predict_proba(X)) 
        a = [A[i, y[i]] for i in range(self.n)]
        rf = []
        for m in range(self.K): 
            rf.append(RandomForestRegressor(max_depth=25, random_state=0))
            xm = numpy.array([X[i] for i in range(self.n) if y[i] == m])
            am = numpy.array([a[i] for i in range(self.n) if y[i] == m])
            rf[-1].fit(xm, am.squeeze())
        return rf    
    
    def prepareData(self, p, x):
        A = self.basicScore(p)
        G = numpy.array([self.rg[i].predict(x) for i in range(self.K)]).transpose()
        h = numpy.array([H(1 - G[i, :]).squeeze() for i in range(len(p))])
        return x, A, G, h
    
    def activation(self, t, eps = .01):
        return t
        #return numpy.log(1 + numpy.exp(t))#-1 + 2 * sigma(eps * t)

    def extractPar(self, theta, Z):
        d = int(len(theta)/2 - 1)
        return [numpy.expand_dims(theta[(d + 1) * i] + self.activation(Z @ theta[1 + (d + 1) * i:(d + 1) * (i + 1)]), axis=-1) for i in [0, 1]]
   
    def bER(self, A, G, h, theta):
        if self.more: xi = self.extractPar(theta, G)
        else: xi = theta
        r = xi[0]**2 + xi[1]**2 * G
        return A * numpy.exp(-r)
    
    def bBase(self, A, G, h, theta):
        return A
    
    def bHR(self, A, G, h, theta):
        H = numpy.expand_dims(h, axis=1)
        if self.more: xi = self.extractPar(theta, G)
        else: xi = theta
        r = xi[0]**2 + xi[1]**2 * H
        return A/(1e-4 + r)
    
    def bTemp(self, A, G, h, theta):
        if self.more: xi = self.extractPar(theta, G)
        else: xi = theta
        r1 = 1 - rescale(1 - A, xi[0]**2)
        r2 = xi[1]**2 * G
        return (r1 + r2)#/sum([xi[i]**2 for i in range(len(xi))])
    
    def bOff(self, A, G, h, theta):
        if self.more: xi = self.extractPar(theta, G)
        else: xi = theta
        r1 = A + xi[1] * G
        r2 = sum([xi[i]**2 for i in range(len(xi))])
        return r1/r2

    def fitTransformation(self, X, y):
        d1 = self.prepareData(self.classifier.predict_proba(X), X), y
        initial_guess =  .5 * numpy.random.randn(self.npars)
        optimal = minimize(self.ML, initial_guess,
            args=(d1, d1), options={'maxiter': 100}, tol = 1e-4)

        print(self.model,  ': done')
        print('status, nit, theta: ', optimal.success, optimal.nit, optimal.x)
        return optimal.x
 
    def ML(self, theta, d1, d2):
        [X1, A1, G1, h1], y1 = d1
        [X2, A2, G2, h2], y2 = d2
        B1 = self.b(A1, G1, h1, theta)
        b1 = numpy.array([B1[i, y1[i]] for i in range(len(y1))])
        b1 = numpy.expand_dims(b1, axis = 1)
        B2 = self.b(A2, G2, h2, theta)
        b2 = numpy.array([B2[i, y2[i]] for i in range(len(y2))])
        b2 = numpy.expand_dims(b2, axis = 1)
        M = b1 - b2.transpose()
        m = numpy.linalg.norm(M - numpy.diag(numpy.diag(M))) 
        ell = m/len(y1)
        return ell + self.rho * theta @ theta

    def predict(self, x):
        p = self.classifier.predict_proba(x) 
        x, A, G, h = self.prepareData(p, x)
        return self.b(A, G, h, self.theta)
    
    def evaluateCPclass(self, d1, d2, alpha = .1):
        X, y = d1
        X, A, G, h = self.prepareData(self.classifier.predict_proba(X), X)
        B = self.b(A, G, h, self.theta)
        q = quantile([B[i, y[i]] for i in range(len(y))], alpha)
        
        X, y = d2
        X, A, G, h = self.prepareData(self.classifier.predict_proba(X), X)
        B = self.b(A, G, h, self.theta)
        sets = [[m for m in range(self.K) if B[i][m] < q] for i in range(len(y))]
        F1s, valCond, sizeCond, f1Cond= F1score(sets, y)
        corrs = HScorr(1 - A, sets)
        sizes = numpy.sum([len(sets[i]) for i in range(len(sets))])/len(y)
        val = numpy.sum([1 for i in range(len(y)) if y[i] in sets[i]])/len(y)
        print("wsc...")
        wsc = wsc_unbiased_label(X, y, sets, 0,
                delta=0.1, M=1000, test_size=0.75, 
                random_state=2020, verbose=False)
        print("wsc label...")
        wsclabel = wsc_unbiased_label(X, y, sets, 1,
                delta=0.1, M=1000, test_size=0.75, 
                random_state=2020, verbose=False)
        
        return val, sizes, F1s, valCond, sizeCond, f1Cond, corrs, wsc, wsclabel

def classifier(D):
    X, Y = D
    rf = RandomForestClassifier(max_depth=20, random_state=0)
    rf.fit(X, Y.squeeze())
    return rf    

def sigma(t):
    return 1/(numpy.exp(-t) + 1)

def quantile(v, alpha):
    m = int(numpy.ceil((1 - alpha) * (len(v) + 1))) - 1
    v = numpy.sort(v, axis = 0)
    return v[m]

def rescale(V, beta):
    S = numpy.exp(beta * V)
    return  S/numpy.expand_dims(numpy.sum(S, axis=1), axis=1) 

def H(p, eps=.0001):
    return -sum([s * numpy.log(s + eps) for s in p])

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
    valCond, sizeCond, f1Cond = F1CPscore(sets, y)#F2 = (TP + TN)/(FP + FN)
    return F1, min(valCond), max(sizeCond), min(f1Cond)

def HScorr(F, sets):
    sizes = numpy.array([len(x) for x in sets])
    h = numpy.array([H(f) for f in F])
    return (sizes@h)/numpy.sqrt(1e-4 + (sizes@sizes) * (h @ h))

def F1CPscore(sets, y):
    mclasses = max(y) + 1
    TP, FP, TN, FN, nsamples, csize = [[0 for m in range(mclasses)] for k in [0, 1, 2, 3, 4, 5]]

    for i in range(len(sets)):
        interval = sets[i]
        positives = [j for j in range(mclasses) if j in interval]
        negatives = [j for j in range(mclasses) if j not in interval]
        for m in range(mclasses):
            TP[m] = TP[m] + 1 * (m in positives) * (y[i] == m)
            FP[m] = FP[m] + 1 * (m in positives) * (y[i] != m)
            TN[m] = TN[m] + 1 * (m in negatives) * (y[i] !=m)
            FN[m] = FN[m] + 1 * (m in negatives) * (y[i] == m)
            nsamples[m] = nsamples[m] + 1 * (y[i] == m)
            csize[m] = csize[m] + len(positives)* (y[i] == m)
    f1Cond = [2 * TP[m]/(2 * TP[m] + FP[m] + FN[m]) for m in  range(mclasses)]
    valCond = [TP[m]/nsamples[m] for m in range(mclasses)]
    sizeCond = [csize[m]/nsamples[m] for m in range(mclasses)]
    return valCond, sizeCond, f1Cond


def splitData(X, y):
    t = 3 #train, cal, test
    n = [i * int(len(X)/t) for i in range(t + 1)]
    return [[X[n[i] : n[i + 1]], y[n[i] : n[i + 1]]]  for i in range(t)]


#load and random-split the data
numpy.random.seed(23456)
digits = load_digits()
Xall = digits.images
yall = digits.target
Xall = Xall.reshape([len(Xall), 64])

#run experiments
#bNames = ['unweighted', 'ER', 'ER+', 'HR', 'HR+', 'temp', 'temp+', 'off', 'off+']
bNames = ['unweighted', 'ER', 'HR', 'temp',  'off']
rho = 0.001
results = []
for k in range(2):
    alpha = .1
    choice = numpy.random.choice(len(Xall), size=len(Xall), replace=False)
    X, y = Xall[choice], yall[choice]
    train, cal, test = splitData(X, y)
    print('|all data|=', len(choice))
    print('|train|, # of attributes = ', train[0].shape)
    print('|Y| = ', max(train[1])+1)
    
    #train the classifier on the training set
    dataset = train
    rfClass = classifier([dataset[0], dataset[1]])
    print(rfClass)
    X, y = train
    Xt, yt = test
    Xc, yc = cal
    bModels = []
    for ib in range(len(bNames)):
        bModels.append(
                transformScores(rfClass, bNames[ib], X, y, rho))
    
    r = [[] for i in bNames]
    for ib in range(len(bNames)): 
        model = bModels[ib]
        score = model.evaluateCPclass(cal, test, alpha = .1)
        val, sizes, F1s, valCond, sizeCond, f1Cond, corrs, wsc, wsclabel = score
        print(bNames[ib])
        print(bModels[ib].theta)
        print("val, sizes, F1s, valCond, sizeCond, f1Cond, corrs, wsc, wsclabel", score)
        r[ib].append(score)

    results.append(r)
print(results)
date='20240128'
numpy.save('results'+date, results)
results = numpy.load('results'+date+'.npy')
nr = results
means=numpy.mean(nr, axis=0)
stds=numpy.std(nr, axis=0)


scoreNames = ["val", "sizes", "F1s", "valCond", "sizeCond", "f1Cond", "corrs", "wsc", "wsclabel"]
for iScore in range(len(scoreNames)):
    print(scoreNames[iScore])
    for ib in range(len(bNames)):
        print(bNames[ib], end=':')
        print(means[ib,:, iScore], '(', stds[ib,:, iScore],')')



