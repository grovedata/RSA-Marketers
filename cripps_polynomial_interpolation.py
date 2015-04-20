import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

#cripps pink apple data. CSV has 2 colums, 1st is an index of weeks, 1-51
#second is the amount for that week

c2009 = np.loadtxt('/data/cripps2009.csv', delimiter=',')
c2010 = np.loadtxt('/data/cripps2010.csv', delimiter=',')

def f(x):
    """ function to approximate by polynomial interpolation"""
    return x * np.sin(x)

# generate points used to plot // 51 points for 51 weeks in the dataset
x_plot = c2009[:,1]

# generate points and keep a subset of them
x = c2009[:,1]
y = f(x)

# create matrix versions of these arrays
X = x[:, np.newaxis]
X_plot = x_plot[:, np.newaxis]

plt.plot(x_plot, f(x_plot), label="ground truth")
plt.scatter(x, y, label="training points")

for degree in [3, 4, 5]:
    model = make_pipeline(PolynomialFeatures(degree), Ridge())
    model.fit(X, y)
    y_plot = model.predict(X_plot)
    plt.plot(x_plot, y_plot, label="degree %d" % degree)

plt.legend(loc='lower left')

plt.show()
