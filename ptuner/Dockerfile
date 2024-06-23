FROM python:3.9

ADD *.py .

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

ENTRYPOINT ["python", "main.py"]