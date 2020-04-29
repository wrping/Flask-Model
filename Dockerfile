FROM python:3.7

ADD requirements.txt /
RUN pip install -r /requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

ADD . /script_2001
WORKDIR /script_2001

# EXPOSE 9000
CMD [ "python" , "script_2001.py"]
