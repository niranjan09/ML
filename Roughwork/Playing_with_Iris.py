import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
from mpl_toolkits.mplot3d import Axes3D

iris = datasets.load_iris()
col1 = [i[0] for i in iris.data]
col2 = [i[1] for i in iris.data]
col3 = [i[2] for i in iris.data]
col4 = [i[3] for i in iris.data]

fig = plt.figure()
fl1 = fig.add_subplot(221)
fl2 = fig.add_subplot(222)
fl3 = fig.add_subplot(223)
fl4 = fig.add_subplot(224)

fl1.scatter(col1[:50], col2[:50], marker = 'o')
fl1.scatter(col1[50:100], col2[50:100], marker = '+')
fl1.scatter(col1[100:150], col2[100:150], marker = '*')

fl2.scatter(col1[:50], col3[:50], marker = 'o')
fl2.scatter(col1[50:100], col3[50:100], marker = '+')
fl2.scatter(col1[100:150], col3[100:150], marker = '*')

fl3.scatter(col1[:50], col4[:50], marker = 'o')
fl3.scatter(col1[50:100], col4[50:100], marker = '+')
fl3.scatter(col1[100:150], col4[100:150], marker = '*')

fl4.scatter(col2[:50], col3[:50], marker = 'o')
fl4.scatter(col2[50:100], col3[50:100], marker = '+')
fl4.scatter(col2[100:150], col3[100:150], marker = '*')

plt.show()
