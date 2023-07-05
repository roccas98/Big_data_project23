
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
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
from nltk.corpus import wordnet as wn
import contractions

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
KAFKA_TOPIC =  'tweet_retriever'      

nltk.download('vader_lexicon')
nltk.download('averaged_perceptron_tagger')
nltk.download("stopwords")
nltk.download('punkt')
nltk.download('wordnet')


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

# Definisco la funzione per processare i tweet questa utilizza un modello addestrato da noi che include anche il preprocessing
def predict_sentiment2(tweet):
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.corpus import wordnet as wn
    from nltk import pos_tag
    
    # Load the pre-trained model
    model = joblib.load('naive_bayes.joblib')
    Tfidf_vect = pickle.load(open('vectorizer.pickle', "rb"))

    tweet = tweet.decode('utf-8')

    #print("model loaded", type(model),"tweet", type(tweet))
    #time.sleep(5)

    #-----------------preprocessing---------------------
    tweet=contractions.fix(tweet)
    words = word_tokenize(tweet)
    words = [word.lower() for word in words]
    words = [word for word in words if word.isalpha()]  
    words = [w for w in words if not w in stopwords.words('english')]
    tag_map = defaultdict(lambda : wn.NOUN)
    tag_map['J'] = wn.ADJ
    tag_map['V'] = wn.VERB
    tag_map['R'] = wn.ADV
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word,tag_map[tag[0]]) for word,tag in pos_tag(words)]
    #---------------------------------------------------

    # Vectorize the tweet
    tweet= Tfidf_vect.transform([str(words)])
    # Predict sentiment using the model
    sentiment = model.predict(tweet)
    
    #print(sentiment)

    average = sum(sentiment) / len(sentiment)
    if average > 0.5:
        return ("positive")
    else:
        if average == 0.5:
            return("neutral")
        else:
            return("negative")

# Register the UDF
spark.udf.register("predict_sentiment2_udf", predict_sentiment2, StringType())


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