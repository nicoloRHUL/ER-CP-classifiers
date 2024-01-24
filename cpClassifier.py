import numpy
import matplotlib as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_digits
import matplotlib.colors as mcolors
from scipy.optimize import minimize

class transformScores:
    def __init__(self, prob, X, y, obj, rho = 0):
        self.n, self.K = prob.shape
        self.rho = rho 
        self.name = obj
        self.rg = self.approximator(prob, X, y)
        if obj == 'ML': 
            self.obj = self.ML
            self.theta = self.fitTransformation(prob, X, y)
        
        if obj == 'size': 
            self.obj = self.smoothSize
            self.theta = self.fitTransformation(prob, X, y)

        if obj == 'unweighted': 
            self.obj = None
            self.theta = numpy.zeros(10)    

    def basicScore(self, prob):
        return 1 - prob

    def approximator(self, prob, X, y):
        A = self.basicScore(prob)
        a = [A[i, y[i]] for i in range(self.n)]
        rf = []
        for m in range(self.K): 
            rf.append(RandomForestRegressor(max_depth=25, random_state=0))
            xm = numpy.array([X[i] for i in range(self.n) if y[i] == m])
            am = numpy.array([a[i] for i in range(self.n) if y[i] == m])
            rf[-1].fit(xm, am.squeeze())
        return rf    
    
    def prepareData(self, prob, X):
        A = self.basicScore(prob)
        G = numpy.array([self.rg[i].predict(X) for i in range(self.K)]).transpose()
        h = numpy.array([H(1 - G[i, :]).squeeze() for i in range(self.n)])
        return X, A, G, h

    def r(self, t, G, h):
        h = numpy.expand_dims(h, axis=1)
        r1, r2 = [t[0 + i] + 
                t[2 + i] * numpy.power(abs(G), t[4 + i]) + 
                t[6 + i] * numpy.power(abs(h), t[8+ i]) for i in [0, 1]]
        return r1, r2
    
    def b(self, A, G, h, theta):
        r1, r2 = self.r(theta, G, h)
        return A * numpy.exp(-r1) - r2 


    def fitTransformation(self, prob, X, y):
        d1 = self.prepareData(prob, X), y
        initial_guess =  .1 * numpy.random.randn(10)
        optimal = minimize(self.obj, initial_guess,
                args=(d1, d1), options={'maxiter': 1000}, tol = 1e-3)
        print(self.name,  ': done')
        print('status, nit, theta: ', optimal.success, optimal.nit, optimal.x)
        return optimal.x
 
    def ML(self, theta, d1, d2):
        [X1, A1, G1, h1], y1 = d1
        [X2, A2, G2, h2], y2 = d2
        B1 = self.b(A1, G1, h1, theta)
        b1 = numpy.array([B1[i, y1[i]] for i in range(self.n)])
        b1 = numpy.expand_dims(b1, axis = 1)
        B2 = self.b(A2, G2, h2, theta)
        b2 = numpy.array([B2[i, y2[i]] for i in range(self.n)])
        b2 = numpy.expand_dims(b2, axis = 1)
        M = b1 - b2.transpose()
        m = numpy.linalg.norm(M - numpy.diag(numpy.diag(M))) 
        ell = m/self.n
        return ell + self.rho * theta @ theta

    def smoothSize(self, theta, d1, d2, scale = 2):
        [X1, A1, G1, h1], y1 = d1
        [X2, A2, G2, h2], y2 = d2
        beta = 2
        B1 = self.b(A1, G1, h1, theta)
        B2 = self.b(A2, G2, h2, theta)
        b2 = numpy.array([B2[i2, y2[i2]] for i2 in range(self.n)])
        w = b2/sum(b2)
        w = numpy.exp(beta * w) + 1e-4
        w = numpy.diag(w / sum(w))
        B1 = numpy.expand_dims(B1, axis = 2)
        B1 = numpy.transpose(B1, axes = [0, 2, 1])
        b2 = numpy.array(b2)
        b2 = numpy.expand_dims(b2, axis = [0, 2])
        S = numpy.sum(sigma(scale * (-(B1 - b2))), axis = 2)
        ell = 1/self.n * numpy.sum(S @ w)
        return ell + self.rho * theta @ theta

    def predict(self, prob, X):
        X, A, G, h = self.prepareData(prob, X)
        return self.b(A, G, h, self.theta)
    
    def evaluateCPclass(self, d1, d2, classifier, alpha = .1):
        X, y = d1
        X, A, G, h = self.prepareData(classifier.predict_proba(X), X)
        B = self.b(A, G, h, self.theta)
        q = quantile([B[i, y[i]] for i in range(len(y))], alpha)
        
        X, y = d2
        X, A, G, h = self.prepareData(classifier.predict_proba(X), X)
        B = self.b(A, G, h, self.theta)
        sets = [[m for m in range(self.K) if B[i][m] < q] for i in range(len(y))]
        F1 = F1score(sets, y)
        sizes = numpy.sum([len(sets[i]) for i in range(len(sets))])/len(y)
        val = numpy.sum([1 for i in range(len(y)) if y[i] in sets[i]])/len(y)
        return sizes, F1, val

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
    return F1

def splitData(X, y):
    t = 3 #train, cal, test
    n = [i * int(len(X)/t) for i in range(t + 1)]
    return [[X[n[i] : n[i + 1]], y[n[i] : n[i + 1]]]  for i in range(t)]


#load and random-split the data
numpy.random.seed(12345)
digits = load_digits()
Xall = digits.images
yall = digits.target
Xall = Xall.reshape([len(Xall), 64])

#run experiments
results = [[] for i in [0, 1, 2]]
for k in [0, 1, 2, 4, 5]:
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
    objs = ['ML', 'size', 'unweighted']#['size', 'ML', 'unweighted']
    rhos = [0.1, 0, 0]#[0, .01, 0]
    for iobj in range(len(objs)):
        obj = objs[iobj]
        rho = rhos[iobj]
        bModels.append(
                transformScores(rfClass.predict_proba(X), X, y, obj, rho))
    
    for iB in range(len(bModels)):
        model = bModels[iB]
        size, f1, val = model.evaluateCPclass(cal, test, rfClass, alpha = .1)
        print('size, f1, val', size, f1, val)
        results[iB].append([size, f1, val])

results = numpy.array(results)
means = numpy.mean(results/results[-1], axis=1)
std = numpy.std(results/results[-1], axis = 1)

print('average (5 runs) relative size, f1, validity [ml, smoothSize] ', means[:-1])
print('std deviation (5 runs) size, f1, validity [ml, smoothSize]', std[:-1])


plt.errorbar([0, 1], means[:-1, 0], yerr=std[:-1, i], alpha=.5, label='PS average size')
plt.errorbar([0, 1], means[:-1, 1], yerr=std[:-1, i], alpha=.5, label='PS f1')
plt.plot([0, 1], [1, 1], 'k--', alpha=.5, label='unweighted CP')
plt.xticks([0, 1], labels=['ml', 'size'])
plt.legend()
plt.title('relative size and f1 gains')
plt.show()
print(results)

