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
        self.classifier = classifier
        self.n, self.K = self.classifier.predict_proba(X).shape
        self.npars = 2 * (self.K + 1)
        self.rho = rho
        self.model = model
        if model[:2] == 'ER':
            self.b = self.bER
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

    def accumulator(self, f):
        G = [[sum(numpy.sort(f[i])[::-1][:m]) 
        #G = [[sum(numpy.sort(f[i])[:m]) 
            for m in range(1, len(f[0])+1)] for i in range(len(f))]
        return numpy.array(G)
    
    def prepareData(self, p, x):
        p = rescale(p, .001)
        A = self.basicScore(p)
        G = self.accumulator(p)
        return x, A, G

    def bER(self, A, G, theta):
        r = 1 + G@theta
        return numpy.diag(numpy.exp(-(r**2))) @ A
    
    def bBase(self, A, G, theta):
        return A
    
    def bOff(self, A, G, theta):
        r = numpy.expand_dims(G@theta, axis=-1)
        return A - r

    def fitTransformation(self, X, y):
        x, A, G = self.prepareData(self.classifier.predict_proba(X), X)
        g = G.T@G + self.rho * numpy.eye(len(G[0]))
        if self.model == 'ER':
            s = G.T @ numpy.ones(len(G))
            theta = - numpy.linalg.pinv(g) @ s
        if self.model=='off':
            a = numpy.array([A[i, y[i]] for i in range(len(y))])
            s = G.T@ (a-1/2)
            theta = numpy.linalg.pinv(g) @ s
        print(self.model, theta)
        return theta

    def predict(self, x):
        p = self.classifier.predict_proba(x) 
        x, A, G = self.prepareData(p, x)
        return self.b(A, G, self.theta)
    
    def evaluateCPclass(self, d1, d2, alpha = .1):
        X, y = d1
        X, A, G= self.prepareData(self.classifier.predict_proba(X), X)
        B = self.b(A, G, self.theta)
        q = quantile([B[i, y[i]] for i in range(len(y))], alpha)
        
        X, y = d2
        X, A, G = self.prepareData(self.classifier.predict_proba(X), X)
        B = self.b(A, G, self.theta)
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

def quantile(v, alpha):
    m = int(numpy.ceil((1 - alpha) * (len(v) + 1))) - 1
    v = numpy.sort(v, axis = 0)
    return v[m]

def rescale(V, beta):
    S = numpy.exp(beta + V)
    return  S/numpy.expand_dims(numpy.sum(S, axis=1), axis=1) 

def H(v):
    return -sum([q * numpy.log(1e-4 + q) for q in v])

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
numpy.random.seed(1234)
digits = load_digits()
Xall = digits.images
yall = digits.target
Xall = Xall.reshape([len(Xall), 64])

#run experiments
bNames = ['unweighted', 'ER', 'off']
rho = 0.0001
results = []
for k in range(10):
    alpha = .05
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
date='20240131'
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
        print(means[ib,:, iScore][0], '(', stds[ib,:, iScore][0],')')



