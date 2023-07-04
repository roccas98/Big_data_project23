import tweepy
from kafka import KafkaProducer


bootstrap_servers = 'localhost:9092'                                
topic ='tweet_retriever'                                            
producer = KafkaProducer(bootstrap_servers=bootstrap_servers)       
print("Connected to Kafka producer")



class IDPrinter(tweepy.StreamingClient):
    def __init__(self, bearer):
        super().__init__(bearer)
        self.i = 0

    def on_tweet(self, tweet):
        self.i += 1
        aux = tweet.text + " Tweet number: " + str(self.i)
        print(aux)
        producer.send(topic=topic,key=str(self.i).encode('utf-8'),value=tweet.text.encode('utf-8'))
        print("Tweet sent to Kafka")
    

printer = IDPrinter("AAAAAAAAAAAAAAAAAAAAABU6lgEAAAAAkS16sObVmW1wiUQP%2FKA1RTdyeLU%3Ds94lTjjCaThuuFJ7prdD5lF4ZBO54YPWf2qcv99G0xYWiB2tkM")
#printer.delete_rules()

#printer.delete_rules(ids='1660673904619257861')
#printer.delete_rules(ids='1660673904619257861')
#printer.add_rules(tweepy.StreamRule("#freedom -is:retweet lang:eng"))
print(printer.get_rules())
printer.filter()