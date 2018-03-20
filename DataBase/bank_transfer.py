import os
import sys
import pymysql

class transferMoney(object):
    def __init__(self):
        self.conn = conn
    def transfer(self, sourceID, targetID, money):
        # 其他函数若是有错会抛出异常而被检测到
　　　　　try:
            



if __name__ == "__main__":
    if len(sys.argv) >=2:
        sourceID = int(sys.argv[1])
        targetID = int(sys.argv[2])
        money = int(sys.argv[3])

        conn = pymysql.Connect(host='127.0.0.1', port=3306, user='root', passwd="zk1991zk", db="mytest", charset="utf8")
        trMoney = transferMoney(conn)

        try:
            trMoney.transfer(sourceID, targetID, money)

        except Exception as e:
            print("出现问题" + str(e))
        finally:
            conn.close()