import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
from sklearn import feature_extraction, model_selection, naive_bayes, metrics, svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support as score

# Import the dataset using pandas
dataset = pd.read_csv("./spam.csv")

# Remove the unnamed columns
dataset = dataset.drop('Unnamed: 2', 1)
dataset = dataset.drop('Unnamed: 3', 1)
dataset = dataset.drop('Unnamed: 4', 1)

# Rename the columns
dataset = dataset.rename(columns={'v1': 'label', 'v2': 'message'})
# print(dataset.head())

"""
Grouping by the label column we find that there are 747 instances of spam and majority of them contain 'Please call our customer service representative'
while the ham label contains 4825 instances 4516 which are unique and having 'Sorry, I'll call later' as the most frequent 
"""
# print(dataset.groupby('label').describe())


#Counts the spams and plots a graph
count_Class = pd.value_counts(dataset["label"], sort=True)
count_Class.plot(kind='bar', color=["green", "red"])
plt.title('Bar Plot')
"""
Displays a graph showing the distibution of spam and not spam instances

"""
# plt.show()

f = feature_extraction.text.CountVectorizer(stop_words='english')
"""
The count vectorizer tokenizes the words in the dataset. It extracts certain words from the sentences from the provided dataset
"""
X = f.fit_transform(dataset["message"])

print(X)


# Classify the spam and non spam messages as 1 and 0 respectively
dataset["label"] = dataset["label"].map({'spam': 1, 'ham': 0})

# Using a test size of 70%
X_train, X_test, y_train, y_test = model_selection.train_test_split(
    X, dataset['label'], test_size=0.70, random_state=42)


list_alpha = np.arange(1/100000, 20, 0.11)
score_train = np.zeros(len(list_alpha))
score_test = np.zeros(len(list_alpha))
recall_test = np.zeros(len(list_alpha))
precision_test = np.zeros(len(list_alpha))

"""
Creates the index for the lists that contain the models scores
"""
count = 0

for alpha in list_alpha:
    bayes = naive_bayes.MultinomialNB(alpha=alpha)
    bayes.fit(X_train, y_train)
    score_train[count] = bayes.score(X_train, y_train)
    score_test[count] = bayes.score(X_test, y_test)
    recall_test[count] = metrics.recall_score(y_test, bayes.predict(X_test))
    precision_test[count] = metrics.precision_score(
        y_test, bayes.predict(X_test))
    count = count + 1

matrix = np.matrix(np.c_[list_alpha, score_train,
                   score_test, recall_test, precision_test])

#Create a dataframe containing 
models = pd.DataFrame(data=matrix, columns=[
                      'alpha', 'Train Accuracy', 'Test Accuracy', 'Test Recall', 'Test Precision'])
print(models.head(n=10))


"""
Get the largest value in the models dataframe Test Precision Column
"""
best_index = models['Test Precision'].idxmax()
print(models.iloc[best_index, :])

rf = RandomForestClassifier(n_estimators=100,max_depth=None,n_jobs=-1)
rf_model = rf.fit(X_train,y_train)

y_pred=rf_model.predict(X_test)
precision,recall,fscore,support =score(y_test,y_pred,pos_label=1, average ='binary')
print('Precision : {} / Recall : {} / fscore : {} / Accuracy: {}'.format(round(precision,3),round(recall,3),round(fscore,3),round((y_pred==y_test).sum()/len(y_test),3)))