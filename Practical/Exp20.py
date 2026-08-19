import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

X, y = make_blobs(
    n_samples=600,
    centers=3,
    n_features=2,
    cluster_std=1.0,
    random_state=42
)

X = StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

model = MLPClassifier(
    hidden_layer_sizes=(2, 2),
    activation='identity',
    learning_rate_init=0.01,
    max_iter=2000,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Experiment 20")
print("Accuracy:", accuracy_score(y_test, y_pred))

plt.scatter(X[:, 0], X[:, 1], c=y)
plt.title("Experiment 20 - Multi Class Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
