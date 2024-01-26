import numpy
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_digits
import matplotlib.colors as mcolors
from scipy.optimize import minimize

class transformScores:
    def __init__(self, classifier, model, X, y, obj, rho = 0):
        self.classifier = classifier
        self.n, self.K = self.classifier.predict_proba(X).shape
        self.rho = rho 
        self.name = obj
        self.rg = self.approximator(X, y)
        self.npars = 2 
        self.model = model
        if model == 'ER':
            self.b = self.bER
        if model == 'HR':
            self.b = self.bHR
        if model == 'temp':
            self.b = self.bTemp
        if model == 'off':
            self.b = self.bOff
        if model == 'unweighted':
            self.b = self.bBase
            self.obj = None
            self.theta = numpy.zeros(self.npars) 
            print(self.name,  ': done')
            print('status, nit, theta: ', None, None, self.theta)

        
        if obj == 'ML' and model != 'unweighted': 
            self.obj = self.ML
            self.theta = self.fitTransformation(X, y)
        
        if obj == 'size' and model != 'unweighted': 
            self.obj = self.smoothSize
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
   
    def bER(self, A, G, h, theta):
        r = theta[0]**2 + theta[1]**2 * G
        return A * numpy.exp(-r)
    
    def bBase(self, A, G, h, theta):
        return A
    
    def bHR(self, A, G, h, theta):
        H = numpy.expand_dims(h, axis=1)
        r = theta[0]**2 + theta[1]**2 * H
        return A/(1e-4 + r)
    
    def bTemp(self, A, G, h, theta):
        r1 = theta[0]**2 * A#1 - rescale(1 - A, theta[0]**2)
        r2 = 1 - rescale(1 - G, theta[1]**2)
        return r1 + r2
    
    def bOff(self, A, G, h, theta):
        r1 = A + theta[1] * G
        r2 = sum(theta**2)
        return r1/r2

    def fitTransformation(self, X, y):
        d1 = self.prepareData(self.classifier.predict_proba(X), X), y
        initial_guess =  .3 * numpy.random.randn(self.npars)
        optimal = minimize(self.obj, initial_guess,
                args=(d1, d1),# method='CG', 
                options={'maxiter': 100}, tol = 1e-4)
        print(self.name, self.model,  ': done')
        print('status, nit, theta: ', optimal.success, optimal.nit, optimal.x)
        return optimal.x
 
    def ML(self, theta, d1, d2):
        [X1, A1, G1, h1], y1 = d1
        [X2, A2, G2, h2], y2 = d2
        B1 = self.b(A1, G1, h1, theta)
        b1 = numpy.array([B1[i, y1[i]] for i in range(len(y1))])
        b1 = numpy.expand_dims(b1, axis = 1)
        B2 = self.b(A2, G2, h2, theta)
        b2 = numpy.array([B2[i, y2[i]] for i in range(len(y1))])
        b2 = numpy.expand_dims(b2, axis = 1)
        M = b1 - b2.transpose()
        m = numpy.linalg.norm(M - numpy.diag(numpy.diag(M))) 
        ell = m/len(y1)
        return ell + self.rho * theta @ theta

    def smoothSize(self, theta, d1, d2, scale = 1):
        z1, y1 = d1
        #[X2, A2, G2, h2]
        z2, y2 = d2
        half = int(len(y2)/2)
        z1, y1 = [x[:half] for x in z1], y1[:half]
        z2, y2 = [x[half:] for x in z2], y2[half:]
        X1, A1, G1, h1 = z1
        X2, A2, G2, h2 = z2
        B2 = self.b(A2, G2, h2, theta)
        b2 = numpy.array([B2[i2, y2[i2]] for i2 in range(len(y2))])
        beta = 1
        w = numpy.diag(rescale(
            numpy.expand_dims(b2, axis=1).transpose(), 
            beta).squeeze())
        B1 = self.b(A1, G1, h1, theta)
        B1 = numpy.expand_dims(B1, axis = 2)
        B1 = numpy.transpose(B1, axes = [0, 2, 1])
        b2 = numpy.array(b2)
        b2 = numpy.expand_dims(b2, axis = [0, 2])
        S = numpy.sum(sigma(- scale *(B1 - b2)/numpy.linalg.norm(B1)), 
            axis = 2)
        ell = 1/len(y1) * numpy.sum(S @ w)
        return ell + self.rho * theta @ theta

    def predict(self, x):
        p = self.classifier.predict_proba(x) 
        x, A, G, h = self.prepareData(p, x)
        return self.b(A, G, h, self.theta)
    
    def evaluateCPclass(self, d1, d2, alpha = .1):
        X, y = d1
        print(len(X))
        X, A, G, h = self.prepareData(self.classifier.predict_proba(X), X)
        B = self.b(A, G, h, self.theta)
        q = quantile([B[i, y[i]] for i in range(len(y))], alpha)
        
        X, y = d2
        X, A, G, h = self.prepareData(self.classifier.predict_proba(X), X)
        B = self.b(A, G, h, self.theta)
        sets = [[m for m in range(self.K) if B[i][m] < q] for i in range(len(y))]
        F1s, F1cps= F1score(sets, y)
        corrs = HScorr(1 - A, sets)
        sizes = numpy.sum([len(sets[i]) for i in range(len(sets))])/len(y)
        val = numpy.sum([1 for i in range(len(y)) if y[i] in sets[i]])/len(y)
        return val, sizes, F1s, F1cps, corrs

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
    F1cp = F1CPscore(sets, y)#F2 = (TP + TN)/(FP + FN)
    return F1, F1cp

def HScorr(F, sets):
    sizes = numpy.array([len(x) for x in sets])
    h = numpy.array([H(f) for f in F])
    return (sizes@h)/numpy.sqrt(1e-4 + (sizes@sizes) * (h @ h))

def F1CPscore(sets, y):
    mclasses = max(y) + 1
    TP, FP, TN, FN = [[0 for m in range(mclasses)] for k in [0, 1, 2, 3]]
    for i in range(len(sets)):
        interval = sets[i]
        positives = [j for j in range(mclasses) if j in interval]
        negatives = [j for j in range(mclasses) if j not in interval]
        for m in range(mclasses):
            TP[m] = TP[m] + 1 * (m in positives) * (y[i] == m)# in positives)
            FP[m] = FP[m] + 1 * (m in positives) * (y[i] != m)# in negatives)
            TN[m] = TN[m] + 1 * (m in negatives) * (y[i] !=m)# in negatives) 
            FN[m] = FN[m] + 1 * (m in negatives) * (y[i] == m)# in positives) 
    F1cp = numpy.prod([2 * TP[m]/(2 * TP[m] + FP[m] + FN[m]) for m in  range(mclasses)])
    return F1cp


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
bNames = ['unweighted', 'ER', 'HR', 'temp', 'off']
objs = ['ML', 'size']#['ML', 'size']#['size', 'ML', 'unweighted']
rhos = [0.01, 0.01]#0.1, 0]#[0, .01, 0]
results = []
for k in range(3):
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
    nSamples = 1000
    X, y = [train[i][:nSamples] for i in [0, 1]]
    Xt, yt = test
    Xc, yc = cal
    bModels = [[] for i in bNames]
    for iobj in range(len(objs)):
        obj = objs[iobj]
        rho = rhos[iobj]
        for ib in range(len(bNames)):
            bModels[ib].append(
                transformScores(rfClass, bNames[ib], X, y, obj, rho))
    
    r = [[] for i in bNames]
    for iobjs in range(len(objs)):
        for ib in range(len(bNames)): 
            model = bModels[ib][iobjs]
            score = model.evaluateCPclass(cal, test, alpha = .1)
            val, size, f1, f1cp, corr = score
            print(bNames[ib], objs[iobjs])
            print(bModels[ib][iobjs].theta)
            print('val, size, f1, f1cp, corr', score)
            r[ib].append(score)

    results.append(r)
print(results)
date='20240126'
numpy.save('results'+date, results)
results = numpy.load('results'+date+'.npy')
nr = results/numpy.expand_dims(results[:, 0, :, :], axis=1)
means=numpy.mean(nr, axis=0)
stds=numpy.std(nr, axis=0)

print('size')
iScore = 1 
for ib in range(1, len(bNames)):
    print(bNames[ib])
    print(means[ib,:, iScore])

print('f1cp')
iScore = 3 
for ib in range(1, len(bNames)):
    print(bNames[ib])
    print(means[ib,:, iScore])

print('corr')
iScore = 4 
for ib in range(1, len(bNames)):
    print(bNames[ib])
    print(means[ib,:, iScore])




