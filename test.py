import numpy as np
import matplotlib.pyplot as plt

x = np.random.rand(100)
y = np.random.rand(100)
t = np.arange(100)

plt.scatter(x, y, cmap=plt.cm.rainbow, c=t)
# get color data of graph
c = plt.gca().collections[0].get_facecolors()
print(c)
