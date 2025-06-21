import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# Load dataset
df = pd.read_csv('data/crop_recommendation.csv')

# Separate features and target label
X = df.drop('label', axis=1)
y = df['label']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model trained successfully! Accuracy: {accuracy:.2f}")

# Save the trained model
with open('model/crop_recommendation_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Model saved as 'model/crop_recommendation_model.pkl'")
