import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("customer_booking.csv", encoding="ISO-8859-1")

X = df.drop(columns=['booking_complete'])
y = df['booking_complete']
X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.2f}")

importances = model.feature_importances_
feat_importances = pd.Series(importances, index=X.columns).nlargest(10)

plt.figure(figsize=(10, 6))
sns.barplot(x=feat_importances.values, y=feat_importances.index, palette="viridis")
plt.title("Top 10 Features")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()
