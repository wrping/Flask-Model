#!/usr/bin/env python 
# -*- coding:utf-8 -*-

db_host = '192.168.1.33'
db_port = 3307
db_user = 'root'
db_password = '123456'
table_name = 'data1'
db_name = 'wrp_db'
import pymysql
con = pymysql.Connect(host=db_host, user=db_user, passwd=db_password, db=db_name, port=db_port)

# 读取
# def read_table(cur, sql_order):  # sql_order is a string
# 	try:
# 		cur.execute(sql_order)  # 多少条记录
# 		data = cur.fetchall()
# 		data_list = []
# 		for i in data:
# 			data_list.append(i)
# 		cols = cur.description
# 		col = []
# 		for i in cols:
# 			col.append(i[0])
# 		frame = pd.DataFrame(data_list)
# 	except:  # , e:
# 		frame = pd.DataFrame()
# 		# print e
# 		# continue
# 	return frame
cur = con.cursor()
sql_order = 'select * from  %s' % table_name
cur.execute(sql_order)
print(cur.fetchall())
# frame = read_table(cur, sql_order)
# #
# # print('frame', frame)