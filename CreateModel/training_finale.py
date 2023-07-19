from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.classification import NaiveBayes, LinearSVC, LogisticRegression, RandomForestClassifier
from pyspark.ml.feature import IDF, Tokenizer, StopWordsRemover
from pyspark.sql.functions import udf
from pyspark.sql.types import IntegerType,StringType, ArrayType
from pyspark.sql import SparkSession
from pyspark.ml.feature import CountVectorizer, VectorIndexer
from pyspark.ml import Transformer
from pyspark.ml.param.shared import HasInputCol, HasOutputCol
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable

import re
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download('wordnet')


# Definizione del Lemmatizer Transformer personalizzato
class Lemmatizer(Transformer, HasInputCol, HasOutputCol, DefaultParamsReadable, DefaultParamsWritable):
    def __init__(self, inputCol, outputCol):
        super(Lemmatizer, self).__init__()
        self._inputCol = inputCol
        self._outputCol = outputCol
        self.lemmatizer = WordNetLemmatizer()

    def setInputCol(self, value):
        return self._set(inputCol=value)

    def getInputCol(self):
        return self._inputCol

    def setOutputCol(self, value):
        return self._set(outputCol=value)

    def getOutputCol(self):
        return self._outputCol

    def _transform(self, dataset):
        lemmatize_udf = udf(lambda words: [self.lemmatizer.lemmatize(word) for word in words if len(word) > 0], ArrayType(StringType()))
        return dataset.withColumn(self.getOutputCol(), lemmatize_udf(dataset[self.getInputCol()]))

# Impostazione del contesto Spark
spark = SparkSession.builder.appName('Sentiment_Analysis_Big_Data')\
        .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.1") \
        .getOrCreate()

# Caricamento dei dati

data = spark.read.csv('sentiment140.csv', header=None, inferSchema=True)

#data = data.sample(fraction= 0.01)

# Rinomina le colonne
data = data.selectExpr('_c0 as label', '_c5 as tweet')

# Trasforma i 4 in 1
data = data.withColumn('label', udf(lambda x: 1 if x == 4 else 0, IntegerType())('label'))

# Funzione per rimuovere le emoji
emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # simboli e punti di codice
                           u"\U0001F680-\U0001F6FF"  # simboli di trasporto e mappe
                           u"\U0001F1E0-\U0001F1FF"  # bandiere (emoji bandiera)
                           u"\u0023-\u0039\u2000-\u206F\u2300-\u23FF\u2600-\u26FF\u2700-\u27BF"  # sequenze di caratteri emoji
                           "]+")
remove_emoji_udf = udf(lambda text: emoji_pattern.sub(r'', text))

# Funzione per rimuovere i link
url_pattern = re.compile(r"http\S+|www\S+")
remove_url_udf = udf(lambda text: url_pattern.sub(r'', text))

# Funzione per rimuovere le menzioni degli utenti
mention_pattern = re.compile(r"@\w+")
remove_mention_udf = udf(lambda text: mention_pattern.sub(r'', text))

# Preprocessing del testo
data = data.withColumn('tweet', remove_emoji_udf('tweet'))
data = data.withColumn('tweet', remove_url_udf('tweet'))
data = data.withColumn('tweet', remove_mention_udf('tweet'))

# Definizione dei transformers della Pipeline
tokenizer = Tokenizer(inputCol='tweet', outputCol='words')
stop_words = StopWordsRemover.loadDefaultStopWords('english')
stop_words_remover = StopWordsRemover(inputCol=tokenizer.getOutputCol(), outputCol='filtered').setStopWords(stop_words)
lemmatizer = Lemmatizer(inputCol=stop_words_remover.getOutputCol(), outputCol='lemmatized')
hashing_tf = CountVectorizer(inputCol=lemmatizer.getOutputCol(), outputCol='tf')
idf = IDF(inputCol=hashing_tf.getOutputCol(), outputCol='features')

feature_indexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures")

# Creazione del modello
#nb = NaiveBayes(smoothing=1.0, modelType='multinomial')

# Creazione del modello SVM
#svm = LinearSVC(maxIter=10, regParam=0.1)

# Creazione del modello Logistic Regression
lr = LogisticRegression(maxIter=10)

#rf = RandomForestClassifier(labelCol="label", featuresCol="indexedFeatures", numTrees=10)

(trainingData, testData) = data.randomSplit([0.8, 0.2])

# Creazione della Pipeline
#pipeline = Pipeline(stages=[tokenizer,stop_words_remover, lemmatizer,hashing_tf, idf, nb])
#pipeline = Pipeline(stages=[tokenizer,stop_words_remover, lemmatizer,hashing_tf, idf, svm])
pipeline = Pipeline(stages=[tokenizer,stop_words_remover, lemmatizer,hashing_tf, idf, lr])
#pipeline = Pipeline(stages=[tokenizer,stop_words_remover, lemmatizer,hashing_tf, idf, feature_indexer, rf])

# Salva in una variabile il nome del modello da utilizzare
#model_name = 'NaiveBayes'
#model_name = 'SVM'
model_name = 'LogisticRegression'
#model_name = 'RandomForest'

# Addestramento del modello
model = pipeline.fit(trainingData)

# Valutazione del modello
predictions = model.transform(testData)
evaluate = BinaryClassificationEvaluator(rawPredictionCol='prediction', labelCol='label')

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

ytrue = predictions.select(['label']).toPandas()
ypredict = predictions.select(['prediction']).toPandas()
confusion_matrix = confusion_matrix(ytrue, ypredict)
accuracy = evaluate.evaluate(predictions)

# Stampa dei risultati
classification_report = classification_report(ytrue, ypredict, target_names=['Negative', 'Positive'], digits=4)
print("\nConfusion Matrix:\n", confusion_matrix)
print("\nClassification Report:\n", classification_report)
print('Accuracy:', accuracy)

# Salvataggio dell'accuracy, della confusion matrix e del classification report su file e infine
# salvataggio della pipeline

with open('accuracy_' + model_name +'.txt', 'w') as f:
    print('Accuracy:', accuracy, file=f)
with open('confusion_matrix_' + model_name +'.txt', 'w') as f:
    print("\nConfusion Matrix:\n", confusion_matrix, file=f)
with open('classification_report_' + model_name +'.txt', 'w') as f:
    print("\nClassification Report:\n", classification_report, file=f)

model.write().overwrite().save('pipeline_' + model_name)



