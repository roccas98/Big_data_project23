from kafka import KafkaProducer
import pandas as pd


bootstrap_servers = 'localhost:9092'                                
topic ='tweet_retriever'                                            
producer = KafkaProducer(bootstrap_servers=bootstrap_servers)       
print("Kafka Producer inizializzato.")

data = pd.read_csv('Tweets.csv')
tweets = data['text']
i = 0

for tweet in tweets:
    key = str(i) 
    value = str(tweet)
    producer.send(topic, key=key.encode('utf-8') , value=value.encode('utf-8'))  # Invia la coppia chiave-valore al topic
    producer.flush()
    i += 1

print("Dati inviati con successo come stream a Kafka.")
