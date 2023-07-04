import requests 
import json
import numpy as np
import time


def SendTweetToFlaskApp (row):
	url = 'http://localhost:5000/data'

	
	data={'key': row.get('key'),
    	'value' : row.get('value'),
    	'sentiment' : row.get('sentiment')}
        
	json_data = json.dumps(data)
	response = requests.post(url, json=json_data)
        
	#print("Sentiment analysis sent to flask",data)
	
data=[{'key': 12,'value' : 22,'sentiment' : 32},{'key': 43,'value' : 53,'sentiment' : 63},{'key': 74,'value' : 84,'sentiment' : 94}]


for element in data:
    SendTweetToFlaskApp(element)
    print("element sent",element)
    time.sleep(5)