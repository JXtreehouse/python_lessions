from flask import Flask
from flask_script import Manager
app = Flask(__name__)
manager = Manager(app)

@app.route('/')
def index():
    return '<h1>hello 镜心</h1>'
@app.route('/user/<name>')
def user(name):
    return '<h1>hello, %s!</h1>' % name


if __name__ == '__main__':
    manager.run()