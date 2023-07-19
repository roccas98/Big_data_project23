from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType, ArrayType
from pyspark.ml import Transformer
from pyspark.ml.param.shared import HasInputCol, HasOutputCol
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.ml import PipelineModel
import re
import requests 
import json
import numpy as np
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer



class Lemmatizer(Transformer, HasInputCol, HasOutputCol, DefaultParamsReadable, DefaultParamsWritable):
    def __init__(self, inputCol, outputCol):
        super(Lemmatizer, self).__init__()
        self._inputCol = "filtered"
        self._outputCol = "lemmatized"
        self.lemmatizer = WordNetLemmatizer()

    def __init__(self):
        super(Lemmatizer, self).__init__()
        self._inputCol = "filtered"
        self._outputCol = "lemmatized"
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



KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
KAFKA_TOPIC =  'tweet_retriever'      

spark = SparkSession.builder.appName("Spark_Consumer_Tweet").getOrCreate()
    #togliere il commento per compilare da ide
    #.config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.0") \
    #.getOrCreate()


def SendTweetToFlaskApp (key,value,sentiment):
	url = 'http://localhost:5000/data'
	
	data={'key': key,
    	'value' : value,
    	'sentiment' : sentiment}
        
	json_data = json.dumps(data)
	response = requests.post(url, json=json_data)

# Connessione a kafka
df = spark.readStream.format("kafka") \
    .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP_SERVERS) \
    .option("subscribe", KAFKA_TOPIC) \
    .option("startingOffsets", "earliest") \
    .load()


# Carico il modello addestrato dalla directory
model = PipelineModel.load('models/pipeline_SVM')

def process_row(df, epoch_id):
    global model
    # Converte il DataFrame in una lista di righe
    rows = df.collect()
    
    for row in rows:
        
        key = row.key.decode("utf-8")
        print(key)
        value = row.value.decode("utf-8")
        print(value)
        tweet = [(value,)]

        tweet = spark.createDataFrame(tweet, ["tweet"])

        # Preprocessing del testo
        tweet = tweet.withColumn('tweet', remove_emoji_udf('tweet'))
        tweet = tweet.withColumn('tweet', remove_url_udf('tweet'))
        tweet = tweet.withColumn('tweet', remove_mention_udf('tweet'))

        # Predizione del sentimento
        prediction = model.transform(tweet)
        prediction = prediction.select("prediction").rdd.flatMap(lambda x: x).collect()

        if prediction[0] == 1.0:
            sentiment= "positive"
        else:   
            sentiment="negative"

        # Invio tramite la funzione Chiave, Tweet e Predizione a flask
        SendTweetToFlaskApp(key, value, sentiment)

# Per ogni batch di dati ricevuti da Kafka, esegue la funzione process_row
query = df.writeStream.foreachBatch(lambda df, epoch_id: process_row(df, epoch_id)).start()
query.awaitTermination()