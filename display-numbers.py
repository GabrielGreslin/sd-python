# Import library
from numpy import genfromtxt
import numpy as np
import matplotlib.pyplot as plt

# Read data
data = genfromtxt('data/Optical character recognition/optdigits.tra', delimiter=',', dtype=None)
def create_tab(t):
    s = 8
    r = []
    for i in range(0,8):
        r.append(t[i*s:(i+1)*s])
    return r
x = [create_tab(l[0:-1]) for l in data]
y = [l[-1] for l in data]

def lineToString(line):
    s = ""
    for e in line:
        s = s + str(e) + " "
    return s

print("x:")
i = 0
for line in x[0]:
    print("#"+str(i)+":"+lineToString(line))
    i+=1
print("y:")
print(y[0])

c = True
i = 0
j = 0
while c:
    if y[i] == j:
        img = np.asarray(x[i])
        plt.subplot(1,10,j+1)
        plt.imshow(img, cmap='Greys', interpolation ='none')
        j += 1
    if j>9:
        c = False
    i += 1
plt.show()
