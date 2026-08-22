import pandas as pd


df = pd.read_csv('train.csv')

survivors = df[df['Survived'] == 1]
non_survivors = df[df['Survived'] == 0]

print("Number of survivors:", len(survivors))
print("Number of non-survivors:", len(non_survivors))

gender_survival = df.groupby('Sex')['Survived'].mean()
print("\nSurvived Rate by Gender:")
print(gender_survival)

ticket_class_survival = df.groupby('Pclass')['Survived'].mean()
print("\nSurvived Rate by Ticket Class:")
print(ticket_class_survival)

df['Deck'] = df['Cabin'].str[0]
deck_survival = df.groupby('Deck')['Survived'].mean()
print("\nSurvived Rate by Deck:")
print(deck_survival)


df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
family_survival = df.groupby('FamilySize')['Survived'].mean()
print("\nSurvived Rate by Family Size:")
print(family_survival)

bins = [0, 12, 18, 60, 100]
labels = ['Child', 'Teen', 'Adult', 'Senior']
df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels)

age_survival = df.groupby('AgeGroup', observed=False)['Survived'].mean() 
print("\nSurvived Rate by Age Group:")
print(age_survival)


df['Age'] = df['Age'].fillna(df['Age'].median())

df['Fare'] = df['Fare'].fillna(df['Fare'].median())

df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

feature_columns = ['Pclass', 'Sex', 'Age', 'Fare', 'FamilySize']

X = df[feature_columns]  # (Inputs)
y = df['Survived']       # (Output)

print("\n--- Features (X) Prepared ---")
print(X.head())

print("\n--- Target (y) Prepared ---")
print(y.head())

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print(f"\nRandom Forest Accuracy: {accuracy * 100:.2f}%")


import numpy as np

importances = model.feature_importances_

feature_importance_df = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': importances
})

feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

print("\n--- Feature Importances ---")
print(feature_importance_df.to_string(index=False))
