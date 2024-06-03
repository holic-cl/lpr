FROM tensorflow/tensorflow:2.12.0

RUN apt-get update && apt-get upgrade -y
RUN apt-get install python3-opencv python3-yaml python3-pandas python3-tesserocr python3-dotenv python3-boto3
RUN pip install pytesseract

WORKDIR /app

CMD python reconocedor_automatico.py --cfg config.yaml
#ADD . /app


