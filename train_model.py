# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import joblib
from sklearn.metrics import accuracy_score, classification_report

# Sample data (can be extended with real-world values)
data = pd.DataFrame({
    'ssc_percentage': [75, 85, 60, 90, 55, 70],
    'hsc_percentage': [80, 88, 65, 92, 50, 68],
    'degree_percentage': [70, 89, 60, 95, 45, 65],
    'entrance_score': [110, 140, 90, 160, 70, 95],
    'label': ['Selected', 'Selected', 'Not Selected', 'Selected', 'Not Selected', 'Not Selected']
})

X = data.drop('label', axis=1)
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier(criterion='entropy', max_depth=4)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

joblib.dump(model, 'admission_model.pkl')
