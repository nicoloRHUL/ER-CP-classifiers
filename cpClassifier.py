import numpy
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_digits
import matplotlib.colors as mcolors
from scipy.optimize import minimize

from functions import *

numpy.random.seed(1234)

#load and random-split the data
digits = load_digits()
X = digits.images
y = digits.target
X = X.reshape([len(X), 64])

alpha = .1
rho = 0.01

allResults = []
nExps = 2
for iExp in range(nExps):

    choice = numpy.random.choice(len(X), size=len(X), replace=False)
    X, y = X[choice], y[choice]
    print('|all data|=', len(choice))
    train, val1, val2, cal, test = splitData(X, y)
    print('|train|, # of attributes = ', train[0].shape)
    print('|Y| = ', max(train[1])+1)
    
    #train the classifier on the training set
    dataset = train
    rfClass = classifier([dataset[0], dataset[1]])

    #train the approximator on the first validation set
    dataset = val1
    allprobs = rfClass.predict_proba(dataset[0])
    ER = sum(rfClass.predict(dataset[0])!= dataset[1])/len(dataset[1])
    print('classifier ER', ER)
    probs = numpy.array([1 - allprobs[i, dataset[1][i]].squeeze() for i in range(len(allprobs))])
    rg = approximator([dataset[0], probs, dataset[1]])
    
    data = val1, val2, cal, test
    ddata = [prepareData(data[i], rfClass, rg) for i in range(len(data))]    
    choice = numpy.random.choice(len(ddata[0][0]), size=len(ddata[0][0]), replace=False)
    evaluationData =[[numpy.array([x[i] for i in choice]) for x in d] for d in ddata] 
    d1, d2, dcal, dtest = evaluationData

    #random selection of initialization
    best = 100000
    besttheta = .1 * numpy.random.randn(11)
    for n in range(15):
        initial_guess =  .1 * numpy.random.randn(11)
        initial = ML(initial_guess, d1, d2, rho)
        initial2 = smoothSize(initial_guess, d1, d2, rho)
        if initial + initial2 < best:
            best, besttheta = initial + initial2, initial_guess
    intial_guess = besttheta
    initial_zero = numpy.array([0, 0, 0, 0, 0] + [0, 0, 0, 0, 0])

    #starting values
    initialML = ML(initial_guess, d1, d2, rho)
    initialSize = smoothSize(initial_guess, d1, d2, rho)
    initialDirect = direct(initial_guess, d1, d2, alpha)
    starts = [initialML, initialSize, initialDirect]
    print('starts:', starts)
    
    
    #optimization
    resultML = minimize(ML, initial_guess, args=(d1, d2, rho), 
            tol=1e-3, options={'maxiter': 1000})
    print('ml done')
    resultSize = minimize(smoothSize, initial_guess, args=(d1, d2, rho), 
            tol=1e-3, options={'maxiter': 1000})
    print('size done')
    resultDirect = minimize(direct, initial_guess, args=(d1, d2, alpha), 
            tol=1e-3, options={'maxiter': 1000})
    print('direct done')
    
    #final values
    ends = [ML(resultML.x, d1, d2, rho), 
            smoothSize(resultSize.x, d1, d2, rho), 
            direct(resultDirect.x, d1, d2, alpha)]
    names = ['ML', 'size smooth', 'direct', 'initial', 'unweighted']
    #store parameters
    outputs = [resultML.x, resultSize.x, resultDirect.x, initial_guess, initial_zero]
    #evaluate size and f1 for the optimal parameters
    final = [evaluateCPclass(dcal, dtest, o, alpha) for o in outputs]
    allResults.append(final)
    print('ends:', ends)
    
    #print exp results
    print('model: ', names)
    print('size:', end=' ')
    sizes = [numpy.round(x[0], 4) for x in final]
    print(sizes)
    print('F1:', end=' ')
    f1s = [numpy.round(x[1], 4) for x in final]
    print(f1s)
    print('validity:', end=' ')
    vals = [numpy.round(x[2], 4) for x in final]
    print(vals)

#print average results
relativeSizes = []
relativeF1s = []
for final in allResults:
    print('model: ', names)
    print('size:', end=' ')
    ref = final[-1][0]
    sizes = [numpy.round(x[0]/ref, 4) for x in final[:-1]]
    print(sizes)
    ref = final[-1][1]
    print('F1:', end=' ')
    f1s = [numpy.round(x[1]/ref, 4) for x in final[:-1]]
    print(f1s)
    ref = final[-1][2]
    print('validity:', end=' ')
    vals = [numpy.round(x[2]/ref, 4) for x in final[:-1]]
    print(vals)
    relativeSizes.append(sizes)
    relativeF1s.append(f1s)

quantities = relativeSizes, relativeF1s
names = 'sizes', 'f1'
for k in [0, 1]:
    x = numpy.array(quantities[k])
    average  = numpy.mean(x, axis = 0)
    std  = numpy.std(x, axis = 0)
    s = [str(numpy.round(average[i], 4)) +'pm' + str(numpy.round(std[i], 4)) 
            for i in range(len(average))]


