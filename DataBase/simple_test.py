import pymysql
conn = pymysql.Connect(host='127.0.0.1',port=3306,user='root',passwd='zk1991zk',db='mytest',charset='utf8')
cursor = conn.cursor()
sql = "select * from  user"
cursor.execute(sql)
rs = cursor.fetchall()
print('rs:', rs)

for each in rs:
    print(each)
