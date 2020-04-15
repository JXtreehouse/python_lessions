#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/4/13 下午2:29
# @Author : AlexZ33
# @Site : 使用ｌｏｃｕｓｔ作压力测试
# @File : pressureTest.py
# @Software: PyCharm

from locust import HttpLocust, TaskSet, task


def index(l):
    l.client.get("/")


class UserTasks(TaskSet):
    # one can specify tasks like this
    tasks = [index]

    # but it might be convenient to use the @task decorator
    @task
    def page404(self):
        self.client.get("/rank")


class WebsiteUser(HttpLocust):
    """
    locust user class that does requests to the locust web serve running on localhost
    """
    host = "https://www.baidu.com"
    min_wait = 2000
    max_wait = 5000
    task_set = UserTasks
