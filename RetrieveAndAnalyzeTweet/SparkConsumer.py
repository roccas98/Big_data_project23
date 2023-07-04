
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json,expr
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from pyspark.sql.functions import udf

import requests 
import json
import numpy as np
import nltk

import time

#from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession
from nltk.sentiment import SentimentIntensityAnalyzer


import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
#from sklearn.externals import joblib
import joblib
from nltk.stem import WordNetLemmatizer

KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
KAFKA_TOPIC =  'tweet_retriever'                                                                                                # "my_topic2"

nltk.download('vader_lexicon')
nltk.download('stopwords')


spark = SparkSession.builder.appName("Spark_Consumer_Tweet").getOrCreate()
    #togliere il commento per compilare da ide
    #.config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.0") \
    #.getOrCreate()


def SendTweetToFlaskApp (row):
	url = 'http://localhost:5000/data'
	
	data={'key': row.key,
    	'value' : row.value,
    	'sentiment' : row.sentiment}
        
	json_data = json.dumps(data)
	response = requests.post(url, json=json_data)
#registrazione come UDF
spark.udf.register("SendTweetToFlaskApp", SendTweetToFlaskApp, StringType())
	
# Definisco la funzione per processare i tweet (questa è quella finta)
def ProcessTweetLenght(value):
    print("processing tweets")
    return len(str(value))
#LA REGISTRO COME UDF
spark.udf.register("ProcessTweetLenght", ProcessTweetLenght, StringType()) 

# Definisco la funzione per processare i tweet questa utilizza un modello già addestrato dalla libreria nltk
def predict_sentiment(tweet):
    sia = SentimentIntensityAnalyzer()
    

    result=sia.polarity_scores(tweet.decode('utf-8'))['compound'], StringType()
    print(result[0])
    if result[0] > 0.05:
        print("positive")
        return "positive"
    elif result[0] < -0.05:
        print("negative")
        return "negative"
    else:
        print("neutral")
        return "neutral"
#LA REGISTRO COME UDF
spark.udf.register("predict_sentiment_udf", predict_sentiment, StringType())


# Definisco la funzione per processare i tweet questa utilizza un modello addestrato da noi
def predict_sentiment(tweet):
    # Load the pre-trained model
    model = joblib.load('naive_bayes.joblib')
    Tfidf_vect = pickle.load(open('vectorizer.pickle', "rb"))
    print("model loaded", type(model),"tweet", type(tweet.decode('utf-8')))
    time.sleep(5)
    tweet= Tfidf_vect.transform([tweet.decode('utf-8')])
    # Predict sentiment using the model
    sentiment = model.predict(tweet)
    
    if sentiment > 0.5:
        return ("positive")
    else:
        if sentiment == 0.5:
            return("neutral")
        else:
            return("negative")

# Register the UDF
spark.udf.register("predict_sentiment", predict_sentiment, StringType())


#---------------------------------------------
# Reduce logging
spark.sparkContext.setLogLevel("ERROR")

#connessione a kafka
df = spark.readStream.format("kafka") \
    .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP_SERVERS) \
    .option("subscribe", KAFKA_TOPIC) \
    .option("startingOffsets", "earliest") \
    .load()
    
# Applico la logica per determinare il sentiment

#togliere il commento se si vuole usare la funzione finta
#df = df.selectExpr("CAST(key AS STRING)", "CAST(value AS STRING)") \
#    .withColumn("sentiment", expr("ProcessTweet2(value)")) \
#    .groupBy("key") \
#    .agg(expr("COLLECT_LIST(value) AS value"), expr("COLLECT_LIST(sentiment) AS sentiment"))

df = df.withColumn("sentiment", expr("predict_sentiment2_udf(value)"))
        
df = df.groupBy("key").agg(expr("COLLECT_LIST(value) AS value"), expr("COLLECT_LIST(sentiment) AS sentiment"))


# Seleziono le colonne desiderate come output
df = df.selectExpr("CAST(key AS STRING)", "CAST(value AS STRING)", "CAST(sentiment AS STRING)")
df=df.select("key", "value", "sentiment") \


#Scrivo i risultati
query = df.writeStream \
    .outputMode("update") \
    .foreach(SendTweetToFlaskApp) \
    .start()

query.awaitTermination()