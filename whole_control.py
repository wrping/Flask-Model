import tarfile
import json
import requests
from flask import Response, json
from flask import Flask
from flask import request


#创建一个falsk对象
app = Flask(__name__)


@app.route('/test', methods=['GET'])
def test():
    return '<h1>test is ok!</h1>'


@app.route('/update', methods=['GET'])
def download_file():
    '''
    下载tar包解压出py脚本
    '''
    url = request.args.get('url')
    file_name = request.args.get('file_name ')

    #查看当前根目录
    import os
    path = os.getcwd()
    print(path)

    r = requests.get(url)
    with open(r".\%s.tar" % file_name, "wb") as code:
        code.write(r.content)

    #解压tar包
    tar = tarfile.open(r".\%s.tar" % file_name)
    tar.extractall("./temp")
    tar.close()



@app.route('/2001', methods=['POST'])
def parse_request_2001():
    #获取post请求参数
    datas = request.get_json()
    print(datas)

    from script_2001 import character_select
    return_data = character_select(datas)

    return_data = json.dumps(return_data, ensure_ascii=False)
    return Response(return_data,content_type='application/json')



@app.route('/2002', methods=['POST'])
def parse_request_2002():

    #获取post请求参数
    datas = request.get_json()
    print(datas)

    callback_flag = datas["callbackFlag"]
    print("callbackFlag", callback_flag)
    # if callback_flag == 0:
    from script_2002 import regresssion_predict
    return_data = regresssion_predict(datas)
    print(return_data)
    return_data = json.dumps(return_data, ensure_ascii=False)
    # else:
    #     from script_2003 import regresssion_predict
    #     return_data = regresssion_predict()
    #     return_data = json.dumps(return_data, ensure_ascii=False)

    return Response(return_data,content_type='application/json')

@app.route('/2004', methods=['POST'])
def parse_request_2004():

    #获取post请求参数
    datas = request.get_json()
    print(datas)

    callback_flag = datas["callbackFlag"]
    print("callbackFlag", callback_flag)
    # if callback_flag == 0:
    from script_2004 import time_series
    return_data = time_series(datas)
    return_data = json.dumps(return_data, ensure_ascii=False)

    return Response(return_data, content_type='application/json')


if __name__ == "__main__":
    #启动服务
    # app.run()
    # app.run(host="0.0.0.0", port=9000,debug=True)
    app.run(host="192.168.4.35", port=9000)



