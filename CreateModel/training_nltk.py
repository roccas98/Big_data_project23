from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn import linear_model
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from dataset_nltk import get_data
import joblib
import pickle
import time
import sys

# If train data equal to 1, the models are trained and saved
# We used this to save some time
train_data = 1

if train_data == 1:

    data,labels = get_data()


    # Split of the dataset into train and test
    Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(data,labels,test_size=0.3)

    # Label encoding of the labels
    Encoder = LabelEncoder()
    Train_Y = Encoder.fit_transform(Train_Y)
    Test_Y = Encoder.fit_transform(Test_Y)

    # Vectorization of the tweets
    Tfidf_vect = TfidfVectorizer(ngram_range=(1, 2),max_features=5000)
    Tfidf_vect.fit(data)

    # Save the vectorizer
    current_path = sys.path[0]
    filename = current_path + "/vectorizer.pickle"
    pickle.dump(Tfidf_vect, open(filename, "wb"))

    # Transform dei tweet in vettori tf-idf
    data_Tfidf = Tfidf_vect.transform(data)

    # Inizializza il modello (es. Naive Bayes)
    #Naive = naive_bayes.MultinomialNB()

    # Esegui la cross-validation e ottieni le previsioni dei label
    #predicted_labels = cross_val_predict(Naive, data_Tfidf, labels, cv=5)

    # Calcola e stampa il classification report
    #print("Classification Report:\n", classification_report(labels, predicted_labels, digits = 4))

    # Calcola e stampa l'accuracy score
    #print("Accuracy Score:", accuracy_score(labels, predicted_labels))

    # Calcola e stampa la confusion matrix
    #confusion_mat = confusion_matrix(labels, predicted_labels)
    #print("Confusion Matrix:\n", confusion_mat)

    # Save the model
    #current_path = sys.path[0]
    #filename = current_path + "/naive_bayes_mix.joblib"
    #joblib.dump(Naive, filename)

    # Save the classification report in a file
    #current_path = sys.path[0]
    #filename = current_path + "/naive_bayes_classification_report_mix.txt"
    #with open(filename, 'w') as f:
        #print(classification_report(labels, predicted_labels, digits=4), file=f)
    
    # Save the accuracy score in a file
    #current_path = sys.path[0]
    #filename = current_path + "/naive_bayes_accuracy_score_mix.txt"
    #with open(filename, 'w') as f:
        #print(accuracy_score(labels, predicted_labels)*100, file=f)
    
    # Save the confusion matrix in a file
    #current_path = sys.path[0]
    #filename = current_path + "/naive_bayes_confusion_matrix_mix.txt"
    #with open(filename, 'w') as f:
        #print(confusion_matrix(labels, predicted_labels), file=f)


    # Train model with SVM
    SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')

    # Esegui la cross-validation e ottieni le previsioni dei label
    predicted_labels = cross_val_predict(SVM, data_Tfidf, labels, cv=5)

    # Calcola e stampa il classification report
    print("Classification Report:\n", classification_report(labels, predicted_labels, digits = 4))

    # Calcola e stampa l'accuracy score
    print("Accuracy Score:", accuracy_score(labels, predicted_labels))

    # Calcola e stampa la confusion matrix
    confusion_mat = confusion_matrix(labels, predicted_labels)
    print("Confusion Matrix:\n", confusion_mat)

    # Save the model
    current_path = sys.path[0]
    filename = current_path + "/svm_mix.joblib"
    joblib.dump(SVM, filename)

    # Save the classification report in a file
    current_path = sys.path[0]
    filename = current_path + "/svm_classification_report_mix.txt"
    with open(filename, 'w') as f:
        print(classification_report(labels, predicted_labels, digits=4), file=f)
    
    # Save the accuracy score in a file
    current_path = sys.path[0]
    filename = current_path + "/svm_accuracy_score_mix.txt"
    with open(filename, 'w') as f:
        print(accuracy_score(labels, predicted_labels)*100, file=f)
    
    # Save the confusion matrix in a file
    current_path = sys.path[0]
    filename = current_path + "/svm_confusion_matrix_mix.txt"
    with open(filename, 'w') as f:
        print(confusion_matrix(labels, predicted_labels), file=f)
    




