import pymysql

conn = pymysql.Connect(host='127.0.0.1', port=3306, user='root', passwd='zk1991zk',db="mytest")

conn.autocommit(False)
cursor = conn.cursor()

sqlInsert = "insert into user(id, name) value('6','Alex')"
sqlUpdate = "update user set name = 'Jason',email='sha5xiang@gmail.com' where id = '2'"
sqlDelete = "delete from user where id='6' "
try:
    cursor.execute(sqlInsert)
    print(cursor.rowcount)
    cursor.execute(sqlUpdate)
    print(cursor.rowcount)
    cursor.execute(sqlDelete)
    print(cursor.rowcount)

    
    conn.commit()
except Exception as e:
    print("Reason:", e)
    conn.rollback()

cursor.close()
cursor.close()
