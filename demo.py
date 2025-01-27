from flask import Flask
from flask import send_from_directory
from flask import request
from charlstm import *
from predict import predict, sample

app = Flask("Demo") # The constructor takes the name of the app as argument
global model

# This line is a "decorator" which binds the / URL to the hello function
@app.route('/')
def hello():
    return send_from_directory('static', 'index.html')

@app.route('/token', methods=['POST'])
def token():
    token = request.form['token']
    print('token='+token)
    res = sample(model, 20, top_k=3, prime=token)
    print('res='+res)
    if ' ' in res:
        return res.split(' ')[1]
    return res.split(' ')[0]

@app.route('/char', methods=['POST'])
def charnum():
    ch = request.form['char']
    return '%x' % ord(ch)

if __name__ == '__main__':

    with open('rnn_20_epoch.net', 'rb') as f:
        checkpoint = torch.load(f, map_location=torch.device('cpu'))
        
    model = CharRNN(checkpoint['tokens'], n_hidden=checkpoint['n_hidden'], n_layers=checkpoint['n_layers'])
    model.load_state_dict(checkpoint['state_dict'])

    app.run()

