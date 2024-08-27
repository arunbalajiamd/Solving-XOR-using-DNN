import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

model = Sequential([
    Dense(2, input_dim=2, activation='relu'),  
    Dense(1, activation='sigmoid')             
])

model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])


model.fit(X, y, epochs=1000, verbose=0)

def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', marker='o', cmap=plt.cm.RdYlBu)
    plt.xlabel('Input 1')
    plt.ylabel('Input 2')
    plt.title('XOR Problem - Decision Boundary')
    plt.show()

plot_decision_boundary(model, X, y)
