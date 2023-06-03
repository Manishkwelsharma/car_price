FROM python:latest
WORKDIR /usr/app/src
COPY requirements.txt
RUN pip install -r requirements.txt
COPY model.pkl /usr/app/src
COPY templates /usr/app/src/templates
COPY static /usr/app/src/static
COPY app.py ./
CMD ["python", "app.py"]
