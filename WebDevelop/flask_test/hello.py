from flask import Flask
from flask import request

app = Flask(__name__)

@app.route('/')
def index():
    user_agent = request.headers.get('User-Agent')
    return '<h1>hello 镜心的小树屋</h1><br><p>你的浏览器是　%s </p>' %user_agent

if __name__ == '__main__':
    app.run(debug=True)
