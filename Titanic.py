import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('Titanic-Dataset.csv')
df = data.copy()

# Basic info
print(df.info())
print(df.describe())
print(df.isnull().sum())

# Drop columns not useful for EDA
df = df.drop(columns=['Cabin', 'PassengerId'])

# Fill missing values
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Step 1: Count how many people shared the same ticket
df['GroupSize'] = df.groupby('Ticket')['Ticket'].transform('count')

# Step 2: Calculate fare per person
df['FarePerPerson'] = df['Fare'] / df['GroupSize']

# Create a new column for IsAlone
df['IsAlone'] = df['GroupSize'] == 1
df['IsAlone'] = df['IsAlone'].map({True: 1, False: 0})

# # Step 3: Calculate the average fare per person for those who are alone
# avg_fare_alone = df[df['IsAlone']]['FarePerPerson'].mean()
# print(avg_fare_alone)

# Convert to categorical
df['Sex'] = df['Sex'].astype('category')
df['Embarked'] = df['Embarked'].astype('category')

# Survival count
# sns.countplot(data=df, x='Survived', hue='Sex')
# plt.title("Survival Count by Sex")
# plt.xlabel("Survived (0 = No, 1 = Yes)")
# plt.ylabel("Count")
# plt.tight_layout()
# plt.show()

# # Survival by Pclass
# sns.countplot(data=df, x='Pclass', hue='Survived')
# plt.title("Survival by Passenger Class")
# plt.xlabel("Passenger Class")
# plt.ylabel("Count")
# plt.tight_layout()
# plt.show()

# # Age distribution
# sns.histplot(data=df, x='Age', bins=30, kde=True)
# plt.title("Age Distribution of Passengers")
# plt.tight_layout()
# plt.show()

# # Survival by Family Size
# df['FamilySize'] = df['SibSp'] + df['Parch']
# sns.barplot(x='FamilySize', y='Survived', data=df)
# plt.title("Survival Rate by Family Size")
# plt.tight_layout()
# plt.show()
print(df.sort_values(by='FarePerPerson', ascending=False).head(10))