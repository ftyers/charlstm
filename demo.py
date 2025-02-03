from flask import Flask
from flask import send_from_directory
from flask import request
from flask_sock import Sock
from charlstm import *
from predict import predict, sample

app = Flask("Demo") # The constructor takes the name of the app as argument

sock= Sock(app)

app.config['SOCK_SERVER_OPTIONS'] = {'ping_interval': 25}

global model
global MODE

@sock.route('/msg')
def msg_socket(ws):
    current_token = ""
    while True:
        message = ws.receive()
        print('current_token:', current_token)
        print('message:', message)
        if message == ' ':
           current_token = ''
           continue
        current_token += message
        res = sample(model, 20, top_k=3, prime=current_token)
        print('res:', res)
        ws.send({'token':current_token, 'res':res})

# This line is a "decorator" which binds the / URL to the hello function
@app.route('/')
def hello():
    if MODE == 'stream':
        return send_from_directory('static', 'streaming.html')
    return send_from_directory('static', 'index.html')

@app.route('/token', methods=['POST'])
def token():
    token = request.form['token']
    print('token=',token)
    res = sample(model, 20, top_k=3, prime=token)
    print('res=',res)
    if ' ' in res:
        return res.split(' ')[1]
    return res.split(' ')[0]

@app.route('/char', methods=['POST'])
def charnum():
    ch = request.form['char']
    return '%x' % ord(ch)

if __name__ == '__main__':
    MODE = 'post'

    with open('rnn_20_epoch.net', 'rb') as f:
        checkpoint = torch.load(f, map_location=torch.device('cpu'), weights_only=True)
        
    model = CharLSTM(checkpoint['tokens'], n_hidden=checkpoint['n_hidden'], n_layers=checkpoint['n_layers'])
    model.load_state_dict(checkpoint['state_dict'])

    if sys.argv[1] == '-s':
        MODE = 'stream'

    app.run()

