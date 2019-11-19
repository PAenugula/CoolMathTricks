import numpy as np
import matplotlib.pyplot as plt

class LinearModel:
    def __init__(self):
        self.points = []
        self.res = []
        self.m = 0
        self.b = 0

    def predict(self, x):
        return self.m*x + self.b

    def put(self, x, y):
        self.res.append(y)
        self.points.append([x,1])
        self.recompute()

    def recompute(self):
        aTa = np.dot(self.transpose(self.points), self.points)
        aTb = np.dot(self.transpose(self.points), self.res)
        Ainv = np.linalg.inv(aTa)
        m_b = np.dot(Ainv, aTb)
        self.m, self.b = m_b[0], m_b[1]
        return m_b

    def transpose(self, mat):
        n_mat = [[] for x in range(len(mat[0]))]
        for x in range(len(mat)):
            for y in range(len(n_mat)):
                n_mat[y].append(mat[x][y])
        return n_mat


class MultiModel:
    def __init__(self, power=1):
        self.points = []
        self.res = []
        self.power = power
        self.coeffs = [0 for x in range(power)]

    def __repr__(self):
        return "Model of power {0}: {1}".format(self.power, [x[0] for x in self.coeffs])

    def plot(self):
        x = [a[-2] for a in self.points]
        y = [a[0] for a in self.res]
        t = np.arange(min(x)-1, max(x) + 1, 0.2)
        plt.plot(x, y, 'ro')
        plt.plot(t, self.predict(t), 'b--')
        plt.show()

    def predict(self, x):
        sum = 0
        for i in range(self.power, -1, -1):
            sum += (x**i) * (self.coeffs[self.power-i][0])
        return sum

    def load_sets(self, xs, ys):
        for i in range(len(xs)):
            x = xs[i]
            y = ys[i]
            self.res.append([y])
            val = []
            for i in range(self.power, -1, -1):
                val.append(x ** i)
            self.points.append(val)
        self.recompute()

    def put(self, x, y):
        self.res.append([y])
        val = []
        for i in range(self.power, -1, -1):
            val.append(x**i)
        self.points.append(val)
        if len(self.points) > self.power:
            self.recompute()

    def recompute(self):
        aTa = np.dot(self.transpose(self.points), self.points)
        aTb = np.dot(self.transpose(self.points), self.res)
        Ainv = np.linalg.inv(aTa)
        m_b = np.dot(Ainv, aTb)
        self.coeffs = m_b

    def transpose(self, mat):
        n_mat = [[] for x in range(len(mat[0]))]
        for x in range(len(mat)):
            for y in range(len(n_mat)):
                n_mat[y].append(mat[x][y])
        return n_mat
