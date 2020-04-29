FROM python:3.7

ADD requirements.txt /
RUN pip install -r /requirements.txt

ADD . /whole_control
WORKDIR //whole_control

EXPOSE 9000
CMD [ "python" , "/whole_control.py"]
