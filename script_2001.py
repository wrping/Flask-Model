# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import pymysql




def adaptiveLasso(data_dict):
    '''
    Adaptive-Lasso变量选择模型
    :return:
    '''

    # data_dict = get_initial_data()    ############################# 模拟数据来自本地表格，待注销

    data_dict = data_dict.to_dict(orient="split")  #拆解出为data、column、index
    print(data_dict)

    data = data_dict["data"]
    data_index = data_dict["index"]
    data_columns = data_dict["columns"]
    print("数据概览",data_dict, data_index, data_columns)

    data = pd.DataFrame(data, index=data_index, columns=data_columns)
    column_list = [column for column in data]
    # print(column_list)

    # 导入AdaptiveLasso算法，要在较新的Scikit-Learn才有
    from sklearn.linear_model import LassoLars
    model = LassoLars()
    model.fit(data.iloc[:, 0:13], data['Y'])
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

    pass_data = datas["data"]
    print(pass_data)
    print(type(pass_data))

    callback_flag = datas["callbackFlag"]
    print("callbackFlag",callback_flag)

    #################"使用数据库"####################
    print("使用数据库")
    db_info = datas["dbInfo"]
    print("dbInfo", db_info)
    db_addr = datas["dbInfo"]["dbAddr"]
    print("dbAddr", db_addr)
    db_user = datas["dbInfo"]["dbUser"]
    print("dbUser", db_user)
    db_map = datas["dbMap"]
    print("dbMap", db_map)
    '''
    'dbMap': [{'columnName': 'A', 'columnType': 'VARCHAR', 'inputItem': '姓名'}, {'columnName': 'B', 'columnType': 'VARCHAR', 'inputItem': '年龄'}]
    '''
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

    # column = []
    # 依照映射关系替换列索引
    input_item = pd.DataFrame(Map_df[['columnName', 'inputItem']])
    column_name = pd.DataFrame(list(data_input.columns))
    column_name.columns = ['columnname']
    column = pd.merge(column_name, input_item, left_on='columnname', right_on='columnName', how='left')
    # for i in range(len(db_map)):
    #     for j in column_list:
    #         db_column_i =Map_df.loc[i, 'columnName']
    #         if j == db_column_i:
    #             j = Map_df.loc[i, 'inputItem']
    #             column.append(j)
    # column
    data_input.columns = list(column['inputItem'])
    data_input.set_index('date', inplace=True)
    data_input = data_input[['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10', 'X11', 'X12', 'X13', 'Y']]
    print('datainput', data_input)

    return data_input, callback_flag


def character_select(datas):

    #请求数据
    data_input, callback_flag = parse_datas(datas)

    #特征处理后的结果数据
    data_dict, better_column_list = adaptiveLasso(data_input)  # lasso特征选择

    return_data = {"pass_data": data_dict,
                   "display_data": [],
                   "display_data_type": "",
                   "model_score": {},
                   "display_info": [],
                   "if_callback": 2,
                   "args": {"list": better_column_list,
                            "selected_thisTime": {},  # 时间序列模型中，放上p:0,q:1
                            "selected_lastTime": {},
                            "args_display_type": "checkbox",
                            "args_info": "本次操作说明：以下为推荐使用的建模维度，请从中选择至少5项，用于下一步建模"},

                   "return_data_instructions": "if_callback为0时，可继续下一步也可回调，请将该节点的结果data保存，并传给下一节点使用；if_callback为1时，表示强制回调。",

                   "others": ""}
    return return_data

# # #
# if __name__ == "__main__":
# #     # 测试
#     datas={
#         "callbackFlag": "0",
#         "dbInfo": {
#             "dbAddr": "jdbc:mysql://192.168.1.33:3307/bigdata_hpc?characterEncoding=utf8&serverTimezone=UTC",
#             "dbClass": "com.mysql.jdbc.Driver",
#             "dbPassword": "biG@daTa.",
#             "dbType": "02",
#             "dbUser": "root",
#             "dsId": "xj93jf9djwo9jdiwjdwdj9kdejokd3d9",
#             "dsName": "数据质检测试",
#             "operationName": "121",
#             "operationTime": 1583903895000,
#             "sourceType": "02"
#         },
#         "data":'',
#         "dbMap": [{
#             "columnName": "x1",
#             "columnType": "DOUBLE",
#             "inputItem": "X6",
#             "isHost": "0",
#             "name": "年末总人口"
#         }, {
#             "columnName": "x2",
#             "columnType": "DOUBLE",
#             "inputItem": "X12",
#             "isHost": "0",
#             "name": "第三产业与第二产业产值比"
#         }, {
#             "columnName": "x4",
#             "columnType": "DOUBLE",
#             "inputItem": "X13",
#             "isHost": "0",
#             "name": "居民消费水平"
#         }, {
#             "columnName": "x5",
#             "columnType": "DOUBLE",
#             "inputItem": "X4",
#             "isHost": "0",
#             "name": "城镇居民人均可支配收入"
#         }, {
#             "columnName": "x6",
#             "columnType": "DOUBLE",
#             "inputItem": "X7",
#             "isHost": "0",
#             "name": "全社会固定资产投资额"
#         }, {
#             "columnName": "x7",
#             "columnType": "DOUBLE",
#             "inputItem": "X1",
#             "isHost": "0",
#             "name": "社会从业人数"
#         }, {
#             "columnName": "x8",
#             "columnType": "DOUBLE",
#             "inputItem": "X8",
#             "isHost": "0",
#             "name": "地区生产总值"
#         }, {
#             "columnName": "y",
#             "columnType": "DOUBLE",
#             "inputItem": "Y",
#             "isHost": "0",
#             "name": "财政收入"
#         }, {
#             "columnName": "year",
#             "columnType": "INT",
#             "inputItem": "date",
#             "isHost": "1",
#             "name": "年份"
#         }, {
#             "columnName": "x9",
#             "columnType": "DOUBLE",
#             "inputItem": "X10",
#             "isHost": "0",
#             "name": "税收"
#         }, {
#             "columnName": "x10",
#             "columnType": "DOUBLE",
#             "inputItem": "X2",
#             "isHost": "0",
#             "name": "在岗职工工资总额"
#         }, {
#             "columnName": "x11",
#             "columnType": "DOUBLE",
#             "inputItem": "X5",
#             "isHost": "0",
#             "name": "城镇居民人均消费性支出"
#         }, {
#             "columnName": "x12",
#             "columnType": "DOUBLE",
#             "inputItem": "X9",
#             "isHost": "0",
#             "name": "第一产业产值"
#         }, {
#             "columnName": "x13",
#             "columnType": "DOUBLE",
#             "inputItem": "X3",
#             "isHost": "0",
#             "name": "社会消费品零售总额"
#         }, {
#             "columnName": "x3",
#             "columnType": "INT",
#             "inputItem": "X11",
#             "isHost": "0",
#             "name": "居民消费价格指数"
#         }],
#         "tableName": "test_finantial"
# }

#     character_select(datas)



