FROM python:3.9.15-bullseye

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /output

CMD [ "python", "./flair_tagger.py" ]