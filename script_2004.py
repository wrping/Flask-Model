#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima_model import ARIMA
from datetime import datetime
from dateutil.relativedelta import relativedelta


# 后5行作为验证数据，与预测结果对比，计算准确度
predictnum = 5
# 残差延迟个数
lagnum = 12

def db_data(datas):
    '''
    处理数据模块
    :param datas:
    :return:
    '''


    pass_data = datas["data"]
    print(pass_data)
    print(type(pass_data))

    callback_flag = datas["callbackFlag"]
    print("callbackFlag", callback_flag)


    print("使用数据库")
    db_info = datas["dbInfo"]
    print("dbInfo", db_info)

    db_addr = db_info["dbAddr"]
    print("dbAddr", db_addr)
    db_user = db_info["dbUser"]
    print("dbUser", db_user)

    db_map = datas["dbMap"]
    print("dbMap", db_map)

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

    return frame, callback_flag, db_map

# 源数据, 取时间 + 数值，行号做索引
def parse_datas(frame, db_map):

    file_data = np.round(frame, 2)
    column_list = file_data.columns
    Map_df = pd.DataFrame(db_map)
    for i in column_list:
        if i == 'index':
            column_list = column_list.drop([i])
            print(column_list)
    file_data = file_data[column_list]
    column = []
    for i in file_data.columns:
        for j in range(len(Map_df)):
            if Map_df.loc[j, 'columnName'] == i:
                i = Map_df.loc[j, 'inputItem']
                column.append(i)
    file_data.columns = column
    file_data = file_data[['date', 'Y']]
    print('input_data', file_data)

    data = file_data.iloc[: len(file_data) - predictnum]  # 只有平稳性检测 和 白噪声检测 用到该数据
    print(data)
    # 源数据,取数据，时间做索引
    # index_data = file_data.set_index('date', inplace=True)  # dataframe 类型
    train_data = file_data.iloc[: len(file_data) - predictnum]  # 留后5行，取前面的数据作为训练数据
    pred_data = train_data.iloc[- predictnum:]  # 后5行做预测
    # all_index_data = file_data.copy()  # 取所有样本作为作为训练数据,对样本外的时间进行预测
    # all_index_data = all_index_data.reset_index('date', inplace=True)
    index_list = file_data['date']  # 时间索引
    print('index_list', index_list)
    # 判断是季度数据/月度数据
    date_last = datetime.date((list(index_list)[-1]))
    date_pre = datetime.date((list(index_list)[-2]))
    # print(date_last)
    from dateutil import rrule

    step_months = rrule.rrule(rrule.MONTHLY, dtstart=date_pre, until=date_last).count() - 1

    # step_months = 3  #测试项

    date_next_list = []
    for i in range(1, 6):
        if step_months == 1:
            date_next_i = date_last + relativedelta(months=+i)
            date_next_list.append(datetime.strftime(date_next_i, '%Y-%m-%d'))
        elif step_months == 3:
            date_next_i = date_last + relativedelta(months=+3 * i)
            date_next_list.append(datetime.strftime(date_next_i, '%Y-%m-%d'))
    print("待预测的5个月份/季度", date_next_list)

    return file_data, data, train_data, pred_data, step_months, date_next_list






def stationarityTest(data):
    '''
    平稳性检验
    :return:
    '''

    # 平稳性检验
    from statsmodels.tsa.stattools import adfuller as ADF
    k = 0
    xdata = data['Y']
    adf = ADF(xdata) #平稳性检测
    # print(u'原始序列平稳性检测的p值：',adf[1])
    while adf[1] >= 0.05:
        k = k + 1
        adf = ADF(xdata.diff(k).dropna())

    print(u'原始序列经过%s阶差分后归于平稳，p值为%s' % (k, adf[1]))

    return k


def whitenoiseTest(data):
    '''
    白噪声检验
    :return:
    '''

    # 白噪声检验
    from statsmodels.stats.diagnostic import acorr_ljungbox
    j = 0
    xdata = data['Y']
    [[lb], [p]] = acorr_ljungbox(xdata, lags=1)

    # p = 0.6 #测试
    while p < 0.05 :  # p > 0.05就跳出，是白噪声序列
        if_black = 1  # if_black为1，则最终结果就为1
        print(u'%s阶差分序列为非白噪声序列，对应的p值为：%s' % (j, p))
        j = j + 1
        [[lb], [p]] = acorr_ljungbox(xdata.diff(j).dropna(), lags=1)
        if p > 0.05:
            print(u'%s阶差分序列为白噪声序列，对应的p值为：%s' % (j, p))

    else:
        if j == 0:
            print(u'原序列序列为白噪声序列，对应的p值为：%s' % p)
            if_black = 0
        else:
            if_black = 1

    return if_black

def findOptimalpq(train_data, k):
    '''
    得到模型参数
    :return:
    '''

    xdata = train_data['Y']

    from statsmodels.tsa.arima_model import ARIMA

    # 定阶
    # 一般阶数不超过length/10
    pmax = int(len(xdata) / 10)
    qmax = int(len(xdata) / 10)
    # bic矩阵
    bic_matrix = []
    for p in range(pmax + 1):
        tmp = []
        for q in range(qmax + 1):
            try:
                tmp.append(ARIMA(xdata, (p, k, q)).fit().bic)
            except:
                tmp.append(None)
        bic_matrix.append(tmp)

    bic_matrix = pd.DataFrame(bic_matrix)
    # 先用stack展平，然后用idxmin找出最小值位置。
    p, q = bic_matrix.stack().astype('float64').idxmin()
    print(u'BIC最小的p值和q值为：%s、%s' % (p, q))
    #教程按照BIC最小的p值和q值为：0、1，实际BIC最小的p值和q值为： 1、1
    return p, q

def arimaModelCheck(train_data, p,k,q):
    '''
    模型检验ARIMA(p, d, q)  d为差分阶数
    :return:
    '''

    # 0：-6行内的样本进行建模，对样本内的最后5行进行预测
    xdata = train_data['Y']
    # 教程是按照0、1，建立ARIMA(0,1,1)模型，实际是ARIMA(1,1,1)
    # 建立并训练模型
    # arima = ARIMA(xdata, (0, 1, 1)).fit()
    arima = ARIMA(xdata, (p, k, q)).fit()
    # 预测
    xdata_pred = arima.predict(typ='levels')
    print('xdata_pred', xdata_pred)
    # 计算残差
    pred_error = (xdata_pred - xdata).dropna()

    # 对残差进行白噪声检验
    lb, p = acorr_ljungbox(pred_error, lags=lagnum)
    # p值小于0.05，认为是非白噪声。
    h = (p < 0.05).sum()
    # if h > 0:
    #     print(u'模型ARIMA(0,1,1)不符合白噪声检验')
    # else:
    #     print(u'模型ARIMA(0,1,1)符合白噪声检验')

    if h > 0:
        print(u'模型ARIMA(%s,%s,%s)不符合白噪声检验'% (p,k,q))
        model_if_white = 0
    else:
        print(u'模型ARIMA(%s,%s,%s)符合白噪声检验' % (p,k,q))
        model_if_white = 1

    pred = arima.forecast(predictnum)[0]  #预测样本之外的5个时间单位,取其第一行
    print(pred)
    return model_if_white, pred

def calErrors(pred_data, pred):
    '''
    误差计算
    :return:
    '''
    # 参数初始化
    real = pred_data['Y']

    # 计算误差
    abs_ = (pred - real).abs()
    mae_ = abs_.mean()  # mae
    rmse_ = ((abs_ ** 2).mean()) ** 0.5
    mape_ = (abs_ / real).mean()

    result = mae_, rmse_, mape_
    print(u'平均绝对误差为：%0.4f，\n均方根误差为：%0.4f，\n平均绝对百分误差为：%0.6f。' % (mae_, rmse_, mape_))

    #预测结果评价模块
    from sklearn.metrics import r2_score
    y_pred = list(np.round(pred, 2))
    print('pred', y_pred)
    y_true = list(np.round(real))
    print('real', y_true)
    # R2 决定系数（拟合优度），越接近于1，模型越好
    def standard(x):
        x = pd.DataFrame(x)
        data_mean = x.mean()
        data_std = x.std()
        x = (x - data_mean) / data_std  # 数据标准化
        return x

    y_pred = standard(y_pred)
    y_true = standard(y_true)
    r2_score = r2_score(y_pred, y_true)
    R2_score = "%.2f%%" % (abs(r2_score) * 100)
    print('r2_score', R2_score)
    if r2_score > 0.9:
        assess = '很好'
    elif r2_score < 0.8:
        assess = '不好'
    else:
        assess = '良好'
    print('assess', assess)

    return result, R2_score, assess


def time_series(datas):
    from flask import request
    # 获取post请求参数
    # datas = request.get_json()
    # print(datas)
    #请求数据
    # data_input, callback_flag = parse_datas(datas)
    frame, callback_flag, db_map = db_data(datas)
    file_data, data, train_data, pred_data, step_month, date_next_list = parse_datas(frame, db_map)
    k = stationarityTest(data)
    if_black = whitenoiseTest(data)
    p, q = findOptimalpq(train_data, k)
    model_if_white, pred = arimaModelCheck(train_data, p, k, q)
    result, R2_score, assess = calErrors(pred_data, pred)
    #测试数据
    callback_flag = datas['callbackFlag']
    print('callback_flag', callback_flag)
    #前端输入参数
    input_p = datas['data']['selected_thisTime']['p']
    input_q = datas['data']['selected_thisTime']['q']

    # k =stationarityTest()  #平稳性检验，返回差分阶数k
    print('k', k)
    if k <=5: # 自行规定，最多差分5次
        # if_black  = whitenoiseTest()  #白噪声检测，如果if_black=1，即为非白噪声
        if if_black == 1:   #非白噪声序列，需要提取信息
            if callback_flag == 0:
                p, q = findOptimalpq(train_data, k)  #通过计算，获取p,q最合适的值
            else:
                p, q = input_p,input_q  #callback_flag =1时，回调，由前端输入参数
            print('p', p)
            print('q', q)
            model_if_white, pred = arimaModelCheck(train_data, p, k, q)
            if model_if_white == 1:  #残差属于白噪声序列，无需再提取，可进行下一步
                result, R2_score, assess = calErrors(pred_data, pred)
                print(result)

                #对所有样本数据建模，进行样本外预测
                xdata = file_data['Y']

                # 建立并训练模型
                arima = ARIMA(xdata, (p, k, q)).fit()
                predict = arima.forecast(predictnum)[0]  # 预测样本之外的5个时间单位,取其第一行
                print('predict', predict)

                predict_list = []
                for i in range(len(predict)):
                    out_put = {"date":date_next_list[i],"Y":predict[i]}
                    predict_list.append(out_put)
                print("预测下5个月份/季度的数据", predict_list)
                if_callback = 0

            else:  #残差为非白噪声序列，需要重新调整p，q
                predict_list = []
                R2_score=""
                assess=""
                if_callback = 1  # 告诉后端，强制回调，下面的值都为空即可
    else:
        print("注意：该数据不适合建立时间序列模型！")



    return_data = { "pass_data":{},
                    "display_data":predict_list,
                    "display_data_type": "",
                    "model_assess": {"模型评分": R2_score, "模型评价": assess},
                    "if_display": 0,
                    "display_info": ["display_data","model_assess"],

                    "if_callback": if_callback,
                    "args": {"list":["p", "q"],
                             "selected_thisTime": {"p": p,
                                                   "q": q},
                             "selected_lastTime": {},
                            "args_display_type":"select",
                            "args_info": "本次操作说明：需要调整参数p和q的值，步长为1， 取值范围[0,5]。其中，p是自回归(AR)的项数，用来获取自变量；q是移动平均(MA)的项数，为了使其光滑"},

                    "return_data_instructions": "if_callback为0时，可继续下一步也可回调，请将该节点的结果data保存，并传给下一节点使用；if_callback为1时，表示强制回调。",

                    "others": ""}
    print(return_data)
    return return_data


# if __name__ == '__main__':
#
#     datas = {'callbackFlag': '0',
#              'data': {'selected_lastTime': {},
#                       'selected_thisTime': {"p": 0, "q": 1}},
#              'dbInfo': {
#                  'dbAddr': 'jdbc:mysql://192.168.1.33:3307/wrp_db?characterEncoding=utf8&serverTimezone=UTC',
#                  'dbClass': 'com.mysql.jdbc.Driver',
#                  'dbPassword': 'biG@daTa.',
#                  'dbType': '02',
#                  'dbUser': 'root',
#                  'dsId': 'xj93jf9djwo9jdiwjdwdj9kdejokd3d9',
#                  'dsName': '数据质检测试',
#                  'operationName': '121',
#                  'operationTime': 1583724586000,
#                  'sourceType': '02'},
#              'dbMap': [{'columnName': 'DATETIME', 'columnType': 'date', 'inputItem': 'date'},
#                        {'columnName': 'DATA', 'columnType': 'double', 'inputItem': 'Y'}
#                        ],
#              'tableName': 'discdata'}
#
#     frame, callback_flag, Map_df = db_data(datas)
#     print('file_data', frame)
#     file_data, data, train_data, pred_data, step_months, date_next_list = parse_datas(frame)
#     k = stationarityTest()
#     if_black = whitenoiseTest()
#     p, q = findOptimalpq(k)
#     model_if_white, pred = arimaModelCheck(p, k, q)
    #
    # return_data = time_series()
    # print(return_data)

    # file_data, callback_flag, Map_df = parse_datas(datas)
    # print('file_data', file_data)
    # # 源数据, 取时间 + 数值，行号做索引
    # file_data = np.round(file_data, 2)
    # column_list = file_data.columns
    # for i in column_list:
    #     if i == 'index':
    #         column_list = column_list.drop([i])
    #         print(column_list)
    # file_data = file_data[column_list]
    # column = []
    # for i in file_data.columns:
    #     for j in range(len(Map_df)):
    #         if Map_df.loc[j, 'columnName'] == i:
    #             i = Map_df.loc[j, 'inputItem']
    #             column.append(i)
    # file_data.columns = column
    # file_data = file_data[['date', 'Y']]
    # print('input_data', file_data)
    #
    # data = file_data.iloc[: len(file_data) - predictnum]  # 只有平稳性检测 和 白噪声检测 用到该数据
    # print(data)
    # # 源数据,取数据，时间做索引
    # # index_data = file_data.set_index('date', inplace=True)  # dataframe 类型
    # train_data = file_data.iloc[: len(file_data) - predictnum]  # 留后5行，取前面的数据作为训练数据
    # pred_data = train_data.iloc[- predictnum:]  # 后5行做预测
    # all_index_data = file_data.copy()  # 取所有样本作为作为训练数据,对样本外的时间进行预测
    # # all_index_data = all_index_data.reset_index('date', inplace=True)
    # index_list = file_data['date']  # 时间索引
    # print('index_list', index_list)
    # # 判断是季度数据/月度数据
    # date_last = datetime.date((list(index_list)[-1]))
    # date_pre = datetime.date((list(index_list)[-2]))
    # # print(date_last)
    # from dateutil import rrule
    # step_months = rrule.rrule(rrule.MONTHLY, dtstart=date_pre, until=date_last).count() - 1
    #
    # # step_months = 3  #测试项
    #
    # date_next_list = []
    # for i in range(1, 6):
    #     if step_months == 1:
    #         date_next_i = date_last + relativedelta(months=+i)
    #         date_next_list.append(datetime.strftime(date_next_i, '%Y-%m-%d'))
    #     elif step_months == 3:
    #         date_next_i = date_last + relativedelta(months=+3 * i)
    #         date_next_list.append(datetime.strftime(date_next_i, '%Y-%m-%d'))
    # print("待预测的5个月份/季度", date_next_list)



