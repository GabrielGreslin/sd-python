# Import library
from numpy import genfromtxt
from sklearn import tree

# Read data
data = genfromtxt('data/internetAds/ad.data', delimiter=',', dtype=None)
print(data)

# Build classifier
tree.DecisionTreeClassifier()