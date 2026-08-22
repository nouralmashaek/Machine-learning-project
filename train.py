import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load training data
train_df = pd.read_csv('train.csv')

# Load the testing data (missing answers)
test_df = pd.read_csv('test.csv')

# Family size
train_df['FamilySize'] = train_df['SibSp'] + train_df['Parch'] + 1

# missing data
train_df['Age'] = train_df['Age'].fillna(train_df['Age'].median())
train_df['Fare'] = train_df['Fare'].fillna(train_df['Fare'].median())

# text to numbers
train_df['Sex'] = train_df['Sex'].map({'male': 0, 'female': 1})

feature_columns = ['Pclass', 'Sex', 'Age', 'Fare', 'FamilySize']
X_train = train_df[feature_columns]
y_train = train_df['Survived']

test_df['FamilySize'] = test_df['SibSp'] + test_df['Parch'] + 1
test_df['Age'] = test_df['Age'].fillna(train_df['Age'].median())
test_df['Fare'] = test_df['Fare'].fillna(train_df['Fare'].median())
test_df['Sex'] = test_df['Sex'].map({'male': 0, 'female': 1})

X_test = test_df[feature_columns]


# START
model = RandomForestClassifier(n_estimators=100, random_state=42)

# train it (:
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# csv
submission = pd.DataFrame({
    'PassengerId': test_df['PassengerId'],
    'Survived': predictions
})

submission.to_csv('submission.csv', index=False)
print("submission.csv created successfully! Ready to upload to Kaggle.")
