from flask import Flask

app = Flask('bootcamp')

@app.route('/')
def index():
    return 'Flask'
