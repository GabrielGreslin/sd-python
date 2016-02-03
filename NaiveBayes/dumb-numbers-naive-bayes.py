# Import
from sklearn.naive_bayes import GaussianNB
import numpy
from numpy import genfromtxt

# Data
training = genfromtxt('data/Optical character recognition/optdigits.tra', delimiter=',', dtype=None)
tests = genfromtxt('data/Optical character recognition/optdigits.tes', delimiter=',', dtype=None)

# Variables
x = [l[0:-1] for l in training]
y = [l[-1] for l in training]

xt = [l[0:-1] for l in tests]
yt = [l[-1] for l in tests]

# Build model
gnb = GaussianNB()
model = gnb.fit(x, y)
y_pred = model.predict(xt)
s = model.score(xt,yt)

# Score
print('Score:')
print(s)
