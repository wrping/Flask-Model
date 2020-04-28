import pickle
import pandas as pd
import numpy as np
import json
from flask import Response, json
from flask import Flask
from GM11 import GM11
from pandas.core.frame import DataFrame


# 创建一个falsk对象
app = Flask(__name__)


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
        data[i] = data[i].round(2)

    # data[l + ['y']].to_excel(outputfile)
    print(l)
    data = data[l + ['y']]
    print("灰色模型处理后数据","\n", data)
    return data


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
	y_train = data_train['y'].values  # 标签数据

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
	data[u'y_pred'] = model.predict(x) * data_std['y'] + data_mean['y']
	y_pre = data[['y_pred']][:-2]
	y_predict = data[['y_pred']][-2:]
	y_predict = y_predict.to_dict(orient="dict")  # 转换为字典；orient="dict"是按横取数，orient="list"是按列取数
	y_predict = y_predict['y_pred']
	y_list = []
	n = len(y_predict)
	for key, value in y_predict.items():
		y = {"date": key, "predict": value}
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
    if callback_flag == 1:
        selected_thistime = datas["data"]['selected_thisTime']
		# print('selected_thistime', selected_thistime)

    # 判断是不是直接传入数据
    if pass_data != None:
        print("使用中转数据")
        data_list = datas["data"]["dataInput"]["data"]  #data是二维数组列表
        column_list = datas["data"]["dataInput"]["columns"]
        index_list = datas["data"]["dataInput"]["index"]
        print("行索引",index_list)
        print("列索引", column_list)

        # list转dataframe，将中转数据转化为Dataframe，并添加索引
        data_input = DataFrame(data_list,columns=column_list,index =index_list)
        print("列表转Dataframe后的data_input","\n", data_input)

        characters_selected = datas["data"]["selected_lastTime"]  #人工选择后的特征
        character_list = list(characters_selected.values())  #字典的value转为列表
        print("character_list",character_list)

        return data_input, character_list, index_list, callback_flag, selected_thistime

    else:
        print("使用数据库")


@app.route('/2003', methods=['POST'])
def regresssion_predict():
    from flask import request
    #获取post请求参数
    datas = request.get_json()
    print(datas)

    #处理数据模块
    data_input, character_list, index_list, callback_flag, selected_thistime = parse_datas(datas)
    print(data_input)

    #预测年份参数设置
    data_year_1 = index_list[0]
    predict_year_1 = index_list[-1] + 1
    predict_year_2 = predict_year_1 + 1

    #灰色预测模块
    huise_data = huise(data_input,character_list,index_list,data_year_1,predict_year_1,predict_year_2)

    #svr预测模块
    y_pre, predict_list = svr_predict(huise_data, character_list,data_year_1,predict_year_1,predict_year_2)

    #预测结果评价模块
    '''
    score = "95"
    accuracy = "99%"
    assess = "较好"
    '''
    # 评价模型效果指标
    from sklearn.metrics import r2_score

    y_pred = np.round(y_pre, 2)
    print('y_pred', y_pred)
    y_true = huise_data['y'][:-2]
    print('y_true', y_true)

    # R2 决定系数（拟合优度），越接近于1，模型越好
    r2_score = r2_score(y_true, y_pred)
    R2_score = "%.2f%%" % (r2_score * 100)
    print('r2_score', R2_score)

    if r2_score > 0.9:
        assess = '很好'
    elif r2_score < 0.8:
        assess = '不好'
    else:
        assess = '良好'
    print('assess', assess)

    return_data = { "pass_data":{},
                    "display_data":predict_list,
                    "display_data_type": "",
                    "model_score": {"R2决定系数（拟合优度）": R2_score},
                    "model_assess": assess,
                    "if_display": 0,
                    "display_info": ["display_data","model_score","model_assess"],

                    "if_callback": 0,
                    "args": {"list":["SVR","神经网络"],
                             "selected_thisTime": selected_thistime,
                             "selected_lastTime": {
                                 "x2": "x2",
                                 "x3": "x3",
                                 "x4": "x4",
                                 "x5": "x5",
                                 "x7": "x7"
                             },
                            "args_display_type":"select",
                            "args_info": "本次操作说明：根据模型效果，判断是否要调整模型算法"},

                    "return_data_instructions": "if_callback为0时，可继续下一步也可回调，请将该节点的结果data保存，并传给下一节点使用；if_callback为1时，表示强制回调。",

                    "others":""    }

    return_data = json.dumps(return_data, ensure_ascii=False)
    return Response(return_data,content_type='application/json')


if __name__ == "__main__":
    #启动服务
    # app.run()
    # app.run(host="0.0.0.0", port=9000,debug=False)
    app.run(host="192.168.6.45", port=9000)
























