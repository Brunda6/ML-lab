import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

np.random.seed(42)
values = np.random.rand(100)

labels = ['Class1' if x <= 0.5 else 'Class2' for x in values[:50]] + [None]*50
df = pd.DataFrame({'Point': [f'x{i+1}' for i in range(100)], 'Value': values, 'Label': labels})

print(df.head(), "\n\nSummary:\n", df.describe())

df['Value'].hist(bins=20, edgecolor='black')
plt.title("Value Distribution");
plt.xlabel("Value");
plt.ylabel("Frequency");
plt.show()

X_train = df[df.Label.notna()][['Value']]
y_train = df[df.Label.notna()]['Label']
X_test = df[df.Label.isna()][['Value']]
true_labels = ['Class1' if x <= 0.5 else 'Class2' for x in values[50:]]

k_values = [1, 2, 3, 4, 5, 20, 30]
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
    preds = knn.predict(X_test)
    acc = accuracy_score(true_labels, preds) * 100
    print(f"Accuracy for k={k}: {acc:.2f}%")
