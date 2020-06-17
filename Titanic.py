# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

train_df = pd.read_csv('D:/kaggles/train.csv')
# print(type(train_df))
test_df = pd.read_csv('D:/kaggles/test.csv')
combine = [train_df, test_df]# run certain operations on both datasets

train_df.info()
print('_'*40)
test_df.info()

train_df[['Pclass','Survived']].groupby(['Pclass'], as_index= False).mean()\
    .sort_values(by='Survuved', ascending =False)
train_df[['Sex','Survived']].groupby(['Sex'], as_index= False).mean()\
    .sort_values(by='Survived', ascending =False)
train_df[['SibSp','Survived']].groupby(['SibSp'], as_index= False).mean()\
    .sort_values(by='Survived', ascending =False)
train_df[['Parch','Survived']].groupby(['Parch'], as_index= False).mean()\
    .sort_values(by='Survived', ascending =False)

# A histogram chart is useful for analyzing continous numerical variables
# like Age where banding or ranges will help identify useful patterns.
g = sns.FacetGrid(train_df, col='Survived')
g.map(plt.hist, 'Age', bins=20)

# Alpha-value represents the degree of transparency
# 'col and row' draw a univariate plot on each facet
# Aspect indicates the aspect ratio
grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', height=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=0.5, bins=20)
grid.add_legend()

# Palette = color palette
grid = sns.FacetGrid(train_df, row='Embarked' , height= 2.2 ,aspect= 1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette = 'deep')
grid.add_legend()

# Bar chart are positioned over a label that represents a categorical variable
grid = sns.FacetGrid(train_df,row='Embarked',col='Survived', height=2.2,aspect=1.6)
grid.map(sns.barplot,'Sex','Fare', alpha =.5, ci= None)
grid.add_legend()

# Wrangle data-- 1. Drop 'Ticket' 'Cabin' for correcting data
# .shape returns tuple(rows, column)
print('Before', train_df.shape, test_df.shape, combine[0].shape,combine[1].shape)
train_df = train_df.drop(['Ticket', 'Cabin'], axis =1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis =1)
combine = [train_df, test_df]
print('After', train_df.shape, test_df.shape, combine[0].shape,combine[1].shape)

#  The expand=False returns a DataFrame.RegEx '([A-Za-z]+)\.' matches the first word
#  which ends with a dot character within Name feature.
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract('([A-Za-z]+)\.',expand = False)

pd.crosstab(train_df['Title'], train_df['Sex'])
# Replace titles with  a more common name or classify them as Rare
for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()

# mapping categorical titles to numerical ordinal
# fillna(value) replace null value
title_mapping = {'Mr':1, 'Miss':2, 'Mrs':3, 'Master':4, 'Rare': 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

train_df.head()

# The core item in name-title mow is included in col-title
# Now it's safe to drop name from the dataframe
# pandas drop axis (0 = index; 1 = columns)
train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
test_df = test_df.drop(['Name'], axis=1)
combine = [train_df, test_df]
train_df.shape, test_df.shape

# Continue to convert categorical cols into numerical
gender_map = {'female':1, 'male':0}
for dataset in combine:
    dataset['Sex'] = dataset['Sex']. map(gender_map)

train_df.head()

# In terms of guessing missing age values: based on sets of Pclass and Gender combinations,
# use random numbers between mean and standard deviation.
grid = sns.FacetGrid(train_df,row='Pclass', col='Sex', height= 2.2, aspect= 1.6)
grid.map(plt.hist, 'Age', alpha= .5, bins =20)
grid.add_legend()

# Create an empty array
guess_ages = np.zeros((2,3))
guess_ages

for dataset in combine:
    for i in range(0,2):
        for j in range(0,3):
            guess_df = dataset[(dataset['Sex'] == i) & (dataset['Pclass'] ==j+1)]['Age'].dropna()
            age_guess = guess_df.median()
    # Convert random age float to nearest .5 age
            guess_ages[i,j] = int(age_guess/0.5 + 0.5) * 0.5

    for i in range(0,2):
        for j in range(0,3):
            guess_ages[i, j]=\
            dataset.loc[(dataset.Age.isnull())& (dataset.Sex ==i) & (dataset.Pclass ==j+1),'Age']

    dataset['Age'] = dataset['Age'].astype(int)
train_df.head()



