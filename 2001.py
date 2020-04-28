import pickle
import pandas as pd
import numpy as np
import json
from flask import Response, json
from flask import Flask
from GM11 import GM11

from pandas.core.frame import DataFrame


#创建一个falsk对象
app = Flask(__name__)

# import pickle
# with open('save/model.pkl', 'wb') as f:
#     pickle.dump(model, f)
# y_p = model.predict(data.iloc[:, 0:13])
# print(y_p)  # matrix矩阵中是以空格为分割
# return list(y_p)  #强制转化为列表

def get_initial_data():
    '''
    获取表格数据；转化格式
    :return:
    '''
    inputfile = 'data/data1.csv'
    initial_data = pd.read_csv(inputfile)

    data_dict = initial_data.to_dict(orient="split")  #拆解出为data、column、index
    print(data_dict)
    return data_dict



def adaptiveLasso(data_dict):
    '''
    Adaptive-Lasso变量选择模型
    :return:
    '''

    data_dict = get_initial_data()    ############################# 模拟数据来自本地表格，待注销

    data = data_dict["data"]
    data_index = data_dict["index"]
    data_columns = data_dict["columns"]
    # print("数据概览",data_dict, data_index, data_columns)

    data = pd.DataFrame(data, index=data_index, columns=data_columns)
    column_list = [column for column in data]
    # print(column_list)

    # 导入AdaptiveLasso算法，要在较新的Scikit-Learn才有
    from sklearn.linear_model import LassoLars
    model = LassoLars()
    model.fit(data.iloc[:, 0:13], data['y'])
    # print(model.coef_)

    #结果 "[0.  0.  0.05947013  0.  0.01256303  0.  0.32713801  0.  0.  0.  2.88419736  0.  0.  ]"
    #由系数表可知，系数值为零的要剔除，得到特征变量'x3','x5','x7','x11'
    #实际的特征是按照AdaptiveLasso算法结果选取的['x1', 'x2', 'x3', 'x4', 'x5', 'x7']

    coef_list = model.coef_
    better_column_list = []
    n = len(coef_list)
    for i in range(n):
        c = coef_list[i]
        if c > 0:
            column_name = column_list[i]
            better_column_list.append(column_name)


    data_dict =data.to_dict(orient="split")  #拆解出为data、column、index
    # print("字典数据",data_dict)

    return data_dict, better_column_list



def parse_datas(datas):
    '''
    处理数据模块
    :param datas:
    :return:
    '''
    #测试
    # datas = {'dbInfo': {'dbAddr': 'jdbc:mysql://192.168.1.32:23308/bigdata_hpc?characterEncoding=utf8&serverTimezone=UTC',
    #                     'dbClass': 'com.mysql.jdbc.Driver',
    #                     'dbPassword': '',
    #                     'dbType': '02',
    #                     'dbUser': 'root',
    #                     'dsId': 'xj93jf9djwo9jdiwjdwdj9kdejokd3d9',
    #                     'dsName': '数据质检测试',
    #                     'operationName': '121',
    #                     'operationTime': 1583724586000,
    #                     'sourceType': '02'},
    #         'dbMap': [{'columnName': 'A', 'columnType': 'VARCHAR', 'inputItem': '姓名'}, {'columnName': 'B', 'columnType': 'VARCHAR', 'inputItem': '年龄'}],
    #         'tableName': 'test_model_input',
    #         'data': None,
    #         'callbackFlag': None,
    #         'args': None}

    pass_data = datas["data"]
    print(pass_data)
    print(type(pass_data))

    callback_flag = datas["callbackFlag"]
    print("callbackFlag",callback_flag)


    #判断是不是直接传入数据
    if len(pass_data['selected_lastTime']) or len(pass_data['selected_thisTime']) != 0:
        print("使用中转数据")

    else:
        print("使用数据库")
        db_info = datas["dbInfo"]
        print("dbInfo", db_info)

        db_addr = db_info["dbAddr"]
        print("dbAddr", db_addr)
        db_user = db_info["dbUser"]
        print("dbUser", db_user)

        db_map = datas["dbMap"]
        print("dbMap", db_map)
        '''
        dbMap': [{'columnName': 'year', 'columnType': 'bigint', 'inputItem': 'date'}, {'columnName': 'x1', 'columnType': 'double', 'inputItem': 'X1
        '}, {'columnName': 'x2', 'columnType': 'double', 'inputItem': 'X2'}, {'columnName': 'x3', 'columnType': 'double', 'inputItem': 'X3
        '}, {'columnName': 'x4', 'columnType': 'double', 'inputItem': 'X4'}, {'columnName': 'x5', 'columnType': 'double', 'inputItem': 'X5'}, 
        {'columnName': 'x6', 'columnType': 'double', 'inputItem': 'X6'}, {'columnName': 'x7', 'columnType': 'double', 'inputItem': 'X7'}, 
        {'columnName': 'x8', 'columnType': 'double', 'inputItem': 'X8'}, {'columnName': 'x9', 'columnType': 'double', 'inputItem': 'X9'}, 
        {'columnName': 'x10', 'columnType': 'double', 'inputItem': 'X10'}, {'columnName': 'x11', 'columnType': 'double', 'inputItem': 'X11'}, 
        {'columnName': 'x12', 'columnType': 'double', 'inputItem': 'X12'}, {'columnName': 'x13', 'columnType': 'double', 'inputItem': 'X13'}, 
        {'columnName': 'y', 'columnType': 'double', 'inputItem': 'Y'}]
        '''

        import pymysql
        jdbc = db_addr.split('?')[-2]
        db_charset = db_addr.split('?')[-1]
        split = jdbc.split('/')
        ip = jdbc.split('/')[2]
        db_host = ip.split(':')[0]
        print('dbhost', db_host)
        db_port = ip.split(':')[-1]
        db_port = int(db_port)
        print('dbport', db_port)
        db_name = jdbc.split('/')[-1]
        db_user = db_info['dbUser']
        print('dbuser', db_user)
        db_password = db_info['dbPassword']
        print('dbpassword', db_password)
        table_name = datas['tableName']
        print('tablename', table_name)

        # 连接
        con = pymysql.Connect(host=db_host, user=db_user, passwd=db_password, db=db_name, port=db_port)

        # 读取
        def read_table(cur, sql_order):  # sql_order is a string
            try:
                cur.execute(sql_order)  # 多少条记录
                data = cur.fetchall()
                data_list = []
                for i in data:
                    data_list.append(i)
                cols = cur.description
                col = []
                for i in cols:
                    col.append(i[0])
                frame = pd.DataFrame(data_list, columns=col)
            except:  # , e:
                frame = pd.DataFrame()
                # print e
                # continue
            return frame
        cur = con.cursor()
        sql_order = "select * from %s" % table_name
        frame = read_table(cur, sql_order)
        frame = np.round(frame, 2)

        print('frame', frame)

        # 关闭数据库
        con.commit()
        cur.close()
        con.close()
        print('关闭数据库完成！')

        # 处理数据
        Map_df = pd.DataFrame(db_map)
        column_list = frame.columns
        # print(column_list)

        for i in column_list:
            if i == 'index':
                column_list = column_list.drop([i])
                #         columns=frame.columns[1:]
                #     else:
                #         columns = column_list
                # frame = frame[columns]
                # frame
                print(column_list)

        data_input = frame[column_list]

        column = []
        for i in range(len(db_map)):
            for j in column_list:
                if j == Map_df.loc[i, 'columnName']:
                    j = Map_df.loc[i, 'inputItem']
                    column.append(j)
        # column
        data_input.columns = column
        data_input.set_index('date', inplace=True)

        print('datainput', data_input)

        return data_input, callback_flag



@app.route('/2001', methods=['POST'])
def character_select():
    from flask import request
    #获取post请求参数
    datas = request.get_json()
    print(datas)

    #请求数据
    data_input, callback_flag = parse_datas(datas)

    #特征处理后的结果数据
    data_dict, better_column_list = adaptiveLasso(data_input)  # lasso特征选择

    return_data = { "pass_data":data_dict,
                    "display_data":[],
                    "display_data_type": "",
                    "model_score": {"模型评分":""},
                    "model_assess":"",
                    "if_display": 0,
                    "display_info": [],

                    "if_callback": 2,
                    "args": {"list":better_column_list,
                             "selected_thisTime": {},  #时间序列模型中，放上p:0,q:1
                            "selected_lastTime": {},
                            "args_display_type":"checkbox",
                            "args_info": "本次操作说明：以下为推荐使用的建模维度，请从中选择至少5项，用于下一步建模"},

                    "return_data_instructions": "if_callback为0时，可继续下一步也可回调，请将该节点的结果data保存，并传给下一节点使用；if_callback为1时，表示强制回调。",

                    "others":""    }

    return_data = json.dumps(return_data, ensure_ascii=False)
    return Response(return_data,content_type='application/json')

if __name__ == "__main__":
    #启动服务
    # app.run()
    # app.run(host="0.0.0.0", port=9000,debug=False)
    app.run(host="192.168.6.45", port=9000)



