FROM python:3.7

# ADD requirements.txt /
# RUN pip install -r /requirements.txt

RUN pip install pandas
RUN pip install numpy
RUN pip install pymysql
RUN pip install sklearn
RUN pip install json
RUN pip install tarfile
RUN pip install requests
RUN pip install flask
RUN pip install os
RUN pip install pickle
RUN pip install keras
RUN pip install datetime


ADD . /whole_control
WORKDIR /whole_control

EXPOSE 9000
CMD [ "python" , "whole_control.py"]
