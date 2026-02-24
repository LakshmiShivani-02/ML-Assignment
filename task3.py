from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import CategoricalNB
import numpy as np

# Dataset
data = [
    ['sunny', 'hot', 'high', False, '-'],
    ['sunny', 'hot', 'high', True, '-'],
    ['overcast', 'hot', 'high', False, '+'],
    ['rain', 'mild', 'high', False, '+'],
    ['rain', 'cool', 'normal', False, '+'],
    ['rain', 'cool', 'normal', True, '-'],
    ['overcast', 'cool', 'normal', True, '+'],
    ['sunny', 'mild', 'high', False, '-'],
    ['sunny', 'cool', 'normal', False, '+'],
    ['rain', 'mild', 'normal', False, '+'],
    ['sunny', 'mild', 'normal', True, '+'],
    ['overcast', 'mild', 'high', True, '+'],
    ['overcast', 'hot', 'normal', False, '+'],
    ['rain', 'mild', 'high', True, '-']
]

le = LabelEncoder()
data = np.array(data)
for col in range(data.shape[1] - 1):
    data[:, col] = le.fit_transform(data[:, col])

data[:, -1] = le.fit_transform(data[:, -1])  # '+' -> 1, '-' -> 0

X = data[:, :-1].astype(int)
y = data[:, -1].astype(int)

model = CategoricalNB()
model.fit(X, y)

new_data = np.array([[2, 1, 0, 1]])

probabilities = model.predict_proba(new_data)
predicted_class = model.predict(new_data)[0]
print("Class Probabilities:")
for i, prob in enumerate(probabilities[0]):
    print(f"Class {i} ({'+' if i == 1 else '-'}): {prob:.4f}")

print(f"\nPredicted Class: ({'+' if predicted_class == 1 else '-'})")