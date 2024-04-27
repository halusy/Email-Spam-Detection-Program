import pandas as pd
import numpy as np
import matplotlib as mat
from matspy import spy
import spicy
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from io import StringIO

dataset = pd.read_csv('enron_spam_data.csv')

x = dataset['Message'].values
y = dataset['Spam/Ham']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)

cv = CountVectorizer(stop_words='english', max_df=0.90, min_df=.025,  max_features=500)

#

x_train = cv.fit_transform(x_train.astype('U'))

classifier = svm.SVC(kernel='linear')
classifier.fit(x_train, y_train)

x_test = cv.transform(x_test.astype('U'))

'''
Confusion Matrix Generator:

predictions = classifier.predict(x_test)
cm = confusion_matrix(y_test, predictions, labels=classifier.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classifier.classes_)

disp.plot()
plt.show()
'''

'''
Scatterplot Generator:

d=x_train.todense()
plt.imshow(d,interpolation='none',cmap='binary')
plt.colorbar(ticks=[0,1,2,3,4,5,6,7,8,9,10])
plt.show()
'''

print(classifier.score(x_test, y_test))
print(classifier.coef_.shape[-1])

'''

stillWorking = "yes"

while stillWorking == "yes":

    userEmail = str(input("Enter Email for Spam Testing:"))

    formattedUserEmail = StringIO("""email
    """ + userEmail + """""")

    testingEmail = pd.read_csv(formattedUserEmail, sep=";")
    xnew = testingEmail['email'].values
    testData = cv.transform(xnew.astype('U'))

    isSpam = classifier.predict(testData)

    if isSpam == ['spam']:
        print("\nEmail is Spam")
    if isSpam == ['ham']:
        print('\nEmail is Real')

    stillWorking = str(input("Would you like to check another email? (enter ""yes"" or ""no""): "))
'''