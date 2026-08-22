import pandas as pd
from sklearn.ensemble import RandomForestClassifier

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

combined = pd.concat([train_df, test_df], sort=False)

combined['Title'] = combined['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)

combined['Title'] = combined['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
combined['Title'] = combined['Title'].replace('Mlle', 'Miss')
combined['Title'] = combined['Title'].replace('Ms', 'Miss')
combined['Title'] = combined['Title'].replace('Mme', 'Mrs')

combined['Age'] = combined.groupby('Title')['Age'].transform(lambda x: x.fillna(x.median()))

combined['Fare'] = combined['Fare'].fillna(combined['Fare'].median())

combined['Embarked'] = combined['Embarked'].fillna(combined['Embarked'].mode()[0])

combined['FamilySize'] = combined['SibSp'] + combined['Parch'] + 1

combined['Sex'] = combined['Sex'].map({'male': 0, 'female': 1}).astype(int)

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
combined['Title'] = combined['Title'].map(title_mapping).fillna(0)

embarked_mapping = {'S': 0, 'C': 1, 'Q': 2}
combined['Embarked'] = combined['Embarked'].map(embarked_mapping).astype(int)

train_df = combined[combined['Survived'].notnull()]

test_df = combined[combined['Survived'].isnull()]

feature_columns = ['Pclass', 'Sex', 'Age', 'Fare', 'FamilySize', 'Title', 'Embarked']

X_train = train_df[feature_columns]
y_train = train_df['Survived']
X_test = test_df[feature_columns]

model = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_split=4, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

submission = pd.DataFrame({
    'PassengerId': test_df['PassengerId'].astype(int),
    'Survived': predictions.astype(int)
})

submission.to_csv('submission_v2.csv', index=False)
print("submission_v2.csv created successfully! Ready to upload to Kaggle.")
