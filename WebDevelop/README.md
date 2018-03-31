# 框架
- Django
- flask

flask是一个轻量的web开发应用
示例开发一个小应用
```
from flask import Flask
app = Flask(__name__)
@app.route('/')
def hello_world():
    return 'hello world'
if __name__ == '__main__':
    app.run()
```
将它保存为hello.py，然后用Python解释器来运行，确保你的应用文件名不是flask.py，因为这将与flask本身冲突

```
python hello.py
```

- web.py
- web2py
# 数据库
- mysql
- redis
- mongodb
# 数据处理
- padans
- numpy
- scipy
- sklearn
# 业务框架
- spark
- hadoop
- AWS
- docker

# 后端开发特点
## 技术变更快
- 编程语言
- 行业
- 项目
## 知识面广
- 前端，后端，前后端结合/分离
- 大数据，分布式
- 数据库，关系型/非关系型
- 操作系统，开源项目
## 业务综合
- 设计逻辑
- 实现
- 优化
- 部署（比如支付模块需要独立部署，支付是比较强调安全性，一致性的业务）
# 用python 做后端开发
## 要求
- 熟悉python语言
- 熟悉一款开发工具（我用sublime 和 pycharm）
- 熟悉 一到两种开发框架
- 数据前后端结合、分离技术
## web开发流程


