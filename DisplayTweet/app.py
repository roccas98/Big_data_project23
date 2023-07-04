from flask import Flask, render_template, request, jsonify,json

app = Flask(__name__)

data = [{'key': 1, 'value': 2, 'sentiment': 3},
            {'key': 4, 'value': 5, 'sentiment': 6},
            {'key': 7, 'value': 8, 'sentiment': 9}]

received_data = [{'key': 1, 'value': 2, 'sentiment': 3}]

@app.route('/get_data', methods=['GET'])
def get_data():
    return jsonify(received_data)

@app.route('/data', methods=['POST'])
def receive_data():
    data = request.json
    print("Received data:", data, type(data))
    received_data.append(json.loads(data))
    return jsonify({'status': 'success'})


@app.route('/')
def display_data():
    print("Displaying data:", received_data)
    return render_template('index.html', g={'data': received_data})

if __name__ == '__main__':
    app.run(debug=True)






















'''from flask import Flask, render_template, request, jsonify, g
import ast
import json

app = Flask(__name__)

tweets = []

@app.route('/data', methods=['POST'])
def receive_data():
    data = request.get_json()


    g.data = tweets.append({'key': key, 'value': value, 'sentiment': sentiment})

    return 201


@app.route('/')
def index():
	return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)




tweets = [
    {'key': '1', 'value': 'Hello', 'sentiment': 'positive'},
    {'key': '2', 'value': 'World', 'sentiment': 'negative'},
    {'key': '3', 'value': 'Goodbye', 'sentiment': 'neutral'}
]



    data = json.loads(datastr)
    key = data.get('num')
    value = data.get('tweet')
    sentiment = data.get('sentiment')
    print("Received data: ", data)
'''

# Recupera i dati da qualche parte
	#data = receive_data() 
	#data = {'num': 1, 'tweet': 'ciao', 'sentiment': 'pos'}
	#tweets.append(data)