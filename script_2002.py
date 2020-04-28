
import numpy as np
# from GM11 import GM11
import pickle
from pandas.core.frame import DataFrame


# GM模型，预测
def GM11(x0): #自定义灰色预测函数
   import numpy as np
   import pandas as pd
   x1 = x0.cumsum() #1-AGO序列
   x1 = pd.DataFrame(x1)
   z1 = (x1 + x1.shift())/2.0 #紧邻均值（MEAN）生成序列
   z1 = z1[1:].values.reshape((len(z1)-1,1))  # 转成矩阵
   B = np.append(-z1, np.ones_like(z1), axis = 1)  # 列合并-z1和形状同z1的1值矩阵  19X2
   Yn = x0[1:].reshape((len(x0)-1, 1))  # 转成矩阵 19
   [[a],[b]] = np.dot(np.dot(np.linalg.inv(np.dot(B.T, B)), B.T), Yn) #计算参数，基于矩阵运算，np.dot矩阵相乘，np.linalg.inv矩阵求逆
   f = lambda k: (x0[0]-b/a)*np.exp(-a*(k-1))-(x0[0]-b/a)*np.exp(-a*(k-2)) #还原值
   delta = np.abs(x0 - np.array([f(i) for i in range(1,len(x0)+1)]))  # 残差绝对值序列
   C = delta.std()/x0.std()
   P = 1.0*(np.abs(delta - delta.mean()) < 0.6745*x0.std()).sum()/len(x0)
   return f, a, b, x0[0], C, P #返回灰色预测函数、a、b、首项、方差比、小残差概率

def huise(data,l,index_list,data_year_1,predict_year_1,predict_year_2):
    '''
    地方财政收入灰色预测
    :return:
    '''

    data.loc[predict_year_1] = None
    data.loc[predict_year_2] = None

    data= data.astype('float')   #data中的object类型转为float

    # l = ['x1', 'x2', 'x3', 'x4', 'x5', 'x7']
    # print(l)
    for i in l:
        f = GM11(data[i][np.arange(data_year_1, predict_year_1)].values)[0]
        # 2014年预测结果
        data[i][predict_year_1] = f(len(data) - 1)
        # 2015年预测结果
        data[i][predict_year_2] = f(len(data))
        data[i] = np.round(data[i], 2)

    # data[l + ['Y']].to_excel(outputfile)
    print(l)
    data = data[l + ['Y']]
    # print("灰色模型处理后数据","\n", data)
    return data


def nn_predict(data,feature,data_year_1,predict_year_1,predict_year_2):
    '''
    地方财政收入神经网络预测模型
    :return:
    '''

    input_dim = len(feature)
    modelfile = 'data/1-net.model'  # 模型保存路径

    data_train = data.loc[range(data_year_1, predict_year_1)].copy()  # 取预测年前的数据建模
    data_mean = data_train.mean()
    data_std = data_train.std()
    data_train = (data_train - data_mean) / data_std  # 数据标准化
    x_train = data_train[feature].values  # 特征数据
    y_train = data_train['Y'].values  # 标签数据

    from keras.models import Sequential
    from keras.layers.core import Dense, Activation

    model = Sequential()  # 建立模型
    model.add(Dense(input_dim= input_dim, units=12))
    model.add(Activation('relu'))  # 用relu函数作为激活函数，能够大幅提供准确度
    model.add(Dense(input_dim=input_dim, units=1))
    model.compile(loss='mean_squared_error', optimizer='adam')  # 编译模型
    model.fit(x_train, y_train, epochs=2000, batch_size=16)  # 训练模型，学习一万次
    model.save_weights(modelfile)  # 保存模型参数

    # 预测，并还原结果。
    x = ((data[feature] - data_mean[feature]) / data_std[feature]).values  #z-score 标准化，适用于属性A的最大值和最小值未知的情况
    data[u'y_pred'] = model.predict(x) * data_std['Y'] + data_mean['Y']
    y_pre = data[['y_pred']][:-2]
    y_predict = data[['y_pred']][-2:]
    y_predict = y_predict.to_dict(orient="dict") #转换为字典；orient="dict"是按横取数，orient="list"是按列取数
    y_predict = y_predict['y_pred']
    y_list = []
    for key,value in y_predict.items():
        y = {"date": key,"Y": value}
        y_list.append(y)

    return y_pre, y_list

def svr_predict(data, feature, data_year_1, predict_year_1, predict_year_2):
	'''
	支持向量机回归预测模型
	:return:
	'''

	input_dim = len(feature)
	modelfile = 'data/2-net.model'  # 模型保存路径

	data_train = data.loc[range(data_year_1, predict_year_1)].copy()  # 取预测年前的数据建模
	data_mean = data_train.mean()
	data_std = data_train.std()
	data_train = (data_train - data_mean) / data_std  # 数据标准化
	x_train = data_train[feature].values  # 特征数据
	y_train = data_train['Y'].values  # 标签数据

	from sklearn.svm import SVR

	# 多项式核函数配置支持向量机
	model = SVR(kernel="poly")  # 建立模型
	model.fit(x_train, y_train)  # 训练模型
	# 保存模型参数
	s = pickle.dumps(model)
	f = open(modelfile, "wb+")
	f.write(s)
	f.close()

	# 预测并还原结果
	x = ((data[feature] - data_mean[feature]) / data_std[feature]).values
	data[u'y_pred'] = model.predict(x) * data_std['Y'] + data_mean['Y']
	y_pre = data[['y_pred']][:-2]
	y_predict = data[['y_pred']][-2:]
	y_predict = y_predict.to_dict(orient="dict")  # 转换为字典；orient="dict"是按横取数，orient="list"是按列取数
	y_predict = y_predict['y_pred']
	y_list = []
	for key, value in y_predict.items():
		y = {"date": key, "Y": value}
		y_list.append(y)

	return y_pre, y_list


def parse_datas(datas):
    '''
    处理数据模块
    :param datas:
    :return:
    '''
    pass_data = datas["data"]  #中转数据
    print(pass_data)

    callback_flag = datas["callbackFlag"]
    print("callbackFlag", callback_flag)

    selected_thistime_list = []

    model_type = "神经网络"
    if callback_flag == 0:
        selected_thistime = {"神经网络": "神经网络"}
        selected_lasttime = datas["data"]['selected_lastTime']
    elif callback_flag == None:
        selected_thistime = {"神经网络": "神经网络"}
        selected_lasttime = datas["data"]['selected_lastTime']
    elif callback_flag == 1:
        selected_thistime = datas["data"]['selected_thisTime']
        selected_lasttime = datas["data"]['selected_lastTime']
        for value in selected_thistime.values():
            selected_thistime_list.append(value)
            model_type = selected_thistime_list[0]
    else:
        print("callback_flag参数输入错误！")
    print(model_type)

        #################"使用中转数据"####################
    print("使用中转数据")
    pass_data = datas["data"]["dataInput"]
    data_list = datas["data"]["dataInput"]["data"]  #data是二维数组列表
    column_list = datas["data"]["dataInput"]["columns"]
    index_list = datas["data"]["dataInput"]["index"]
    # print("行索引",index_list)
    # print("列索引", column_list)

    # list转dataframe，将中转数据转化为Dataframe，并添加索引
    data_input = DataFrame(data_list,columns=column_list,index =index_list)
    print("列表转Dataframe后的data_input","\n", data_input)

    characters_selected = datas["data"]["selected_lastTime"]  #人工选择后的特征
    character_list = list(characters_selected.values())  #字典的value转为列表
    print("character_list",character_list)
    print(model_type)
    return pass_data, data_input, character_list, index_list, callback_flag, selected_thistime, selected_lasttime, model_type


def regresssion_predict(datas):
    #处理数据模块
    pass_data, data_input, character_list, index_list, callback_flag, selected_thistime, selected_lasttime,  model_type = parse_datas(datas)
    # print(data_input)

    #预测年份参数设置
    data_year_1 = index_list[0]
    predict_year_1 = index_list[-1] + 1
    predict_year_2 = predict_year_1 + 1

    #灰色预测模块
    huise_data = huise(data_input,character_list,index_list,data_year_1,predict_year_1,predict_year_2)

    y_pre={}
    predict_list={}
    # 判断调用哪个模型
    if callback_flag == 0 :
        y_pre, predict_list = nn_predict(huise_data, character_list, data_year_1, predict_year_1, predict_year_2)
    elif callback_flag == None:
        y_pre, predict_list = nn_predict(huise_data, character_list, data_year_1, predict_year_1, predict_year_2)
    elif callback_flag == 1 :
        if model_type == "神经网络":
            y_pre, predict_list = nn_predict(huise_data, character_list, data_year_1, predict_year_1, predict_year_2)
        elif model_type == "SVR":
            y_pre, predict_list = svr_predict(huise_data, character_list, data_year_1, predict_year_1, predict_year_2)
    else:
        print("callback_flag参数输入错误！")
    print(model_type)
    #神经网络预测模块
    # y_pre, predict_list = nn_predict(huise_data, character_list,data_year_1,predict_year_1,predict_year_2)

    #预测结果评价模块
    from sklearn.metrics import r2_score
    y_pred = np.round(y_pre, 2)
    print('y_pred', y_pred)
    y_true = huise_data['Y'][:-2]
    print('y_true', y_true)
    # R2 决定系数（拟合优度），越接近于1，模型越好
    r2_score = r2_score(y_true, y_pred)
    R2_score = "%.2f%%" % (r2_score * 100)
    print('r2_score', R2_score)
    if r2_score >= 0.9:
        assess = '很好'
    elif r2_score >= 0.8 and r2_score < 0.9:
        assess = '良好'
    elif r2_score >= 0.6 and r2_score < 0.8:
        assess = '较差'
    else:
        assess = '非常差'
    print('assess', assess)

    return_data = { "pass_data":pass_data,
                    "display_data":predict_list,
                    "display_data_type": "",
                    "model_score": {"模型评分": R2_score,"模型评价":assess},
                    "display_info": ["display_data","model_score"],
                    "if_callback": 0,
                    "args": {"list":["SVR","神经网络"],
                             "selected_thisTime": selected_thistime,
                             "selected_lastTime": selected_lasttime,
                            "args_display_type":"select",
                            "args_info": "本次操作说明：根据模型效果，判断是否要调整模型算法"},

                    "return_data_instructions": "if_callback为0时，可继续下一步也可回调，请将该节点的结果data保存，并传给下一节点使用；if_callback为1时，表示强制回调。",

                    "others":""    }
    print(return_data)
    return return_data

# if __name__ == '__main__':
#
#
#
#     request_data ={'data': {'dataInput': {'data': [[435.81, 130.23, 256.6, 4679.61, 109, 512, 948.19, 1121.93, 0.73, 67.67, 470.04, 212.05, 5967.71, 137.45], [498.66, 149.11, 286.9, 5204.29, 103.1, 513.33, 952.59, 1264.63, 0.77, 69.52, 535.02, 223.25, 6608.56, 169.12], [575.86, 146.76, 307.8, 5471.01, 99.5, 508.1, 956.64, 1374.6, 0.86, 74.14, 587.12, 217.97, 7110.54, 186.55], [567.36, 181.62, 325.7, 5851.53, 98.9, 508.14, 959.48, 1500.95, 0.89, 71.14, 657.28, 225.43, 7649.83, 206.89], [608.8, 257.05, 351.7, 6121.07, 99.6, 486.89, 1001.14, 1701.88, 0.88, 73.69, 736.63, 244.47, 8140.55, 244.81], [705.1, 289.52, 382, 6987.22, 101.2, 488.34, 1004.06, 1919.09, 0.92, 78.73, 832.7, 266.73, 8958.7, 304.52], [811.26, 387.66, 399.9, 7191.97, 99.6, 492.61, 1007.18, 2150.76, 0.93, 84.21, 941.36, 290.73, 9337.54, 375.92], [1046.72, 177.56, 431.9, 7867.53, 101, 510.9, 1011.3, 2578.03, 0.86, 89.91, 922.27, 325.41, 10312.91, 451.74], [1258.98, 201.64, 446.42, 8802.44, 102.3, 527.78, 1023.67, 3110.97, 0.78, 105.28, 1044.78, 377.74, 11467.16, 502.17], [1516.84, 270.5, 488.38, 9653.26, 101.5, 542.52, 1043, 3905.64, 0.78, 112.38, 1190.06, 419.8, 13563.32, 725.81], [1849.8, 342.65, 536.05, 10548.05, 101.5, 562.92, 1075, 4462.74, 0.77, 103.35, 1356.79, 487.46, 15476.04, 926.33], [2388.63, 438.36, 588.77, 12028.88, 104.2, 613.93, 1115, 5252.76, 0.78, 110.19, 1603.74, 602.65, 17828.15, 1204.65], [3404.1, 546.26, 657.56, 13422.47, 105.4, 647.32, 1176, 6719.01, 0.78, 122.58, 2078.7, 737.23, 21174.04, 1490.06], [5006.32, 614.27, 717.22, 14801.35, 99, 677.13, 1228.16, 7521.85, 0.85, 128.85, 2430.83, 823.48, 21402.01, 1809.28], [6511.42, 776.65, 816.06, 16561.77, 103.5, 728.7, 1299.29, 9224.46, 0.88, 145.58, 2902.55, 973.84, 24292.6, 2674.4], [7510.67, 1004.51, 894.86, 18424.09, 104.9, 763.16, 1354.58, 11307.28, 0.88, 159.72, 3395.06, 1394.03, 26920.86, 3260.94], [8871.31, 1105.56, 983.87, 20024, 102.7, 803.14, 1413.15, 12893.88, 0.91, 171.6, 3921.43, 1695.89, 29626, 3609.64], [10121.21, 1310.66, 1058.25, 22306, 103.1, 847.46, 1472.21, 14442.01, 0.95, 188.54, 4470.43, 1961.37, 28980, 3904.12], [11654.09, 1486.88, 1116.75, 24290, 101.9, 877.21, 1516.81, 15726.93, 1.01, 199.9, 4738.65, 2063.14, 31506, 4814.4], [13065.18, 1578.07, 1186.59, 26230, 101.7, 896.8, 1546.95, 16538.19, 1.12, 208.82, 5257.28, 2277.96, 34101, 4504.36], [11223.52, 1083.38, 1267.9, 28345, 100.5, 902.42, 1562.12, 17837.89, 1.33, 168.46, 5635.81, 2484.25, 37110, 4828.7], [11274.69, 1075.7, 1349.1, 30284, 100.8, 894.83, 1556.87, 18549.19, 1.42, 168.96, 5729.67, 2556.73, 40278, 4949.66]],
#                                           'columns': ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10', 'X11', 'X12', 'X13', 'Y'],
#                                           'index': [1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015,2016, 2017]},
#                             'selected_thisTime': {'神经网络': '神经网络'},
#                             'selected_lastTime': {'X1': 'X1', 'X2': 'X2', 'X5': 'X5', 'X6': 'X6', 'X7': 'X7', 'X8': 'X8'}},
#                    'dbInfo': {'dsId': 'xj93jf9djwo9jdiwjdwdj9kdejokd3d9',
#                               'dsName': '数据质检测试', 'dbType': '02', 'dbClass': 'com.mysql.jdbc.Driver',
#                               'dbAddr': 'jdbc:mysql://192.168.1.33:3307/bigdata_hpc?characterEncoding=utf8&serverTimezone=UTC',
#                               'dbUser': 'root', 'dbPassword': 'biG@daTa.', 'operationName': '121', 'operationTime': '2020-03-11 05:18:15', 'sourceType': '02'},
#                    'tableName': 'test_finantial', 'dbMap': None,
#                    'callbackFlag': 0, 'args': None}
#
#
#     regresssion_predict(request_data)
#

