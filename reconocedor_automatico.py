import os

# Mostrar solo errores de TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Desabilitar GPU ( correr en CPU )
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from alpr.alpr import ALPR
from argparse import ArgumentParser
import yaml
import logging
from timeit import default_timer as timer
import cv2
import time
from collections import OrderedDict
import pytesseract
import pandas as pd
#pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
import boto3
import re
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
from dotenv import load_dotenv
load_dotenv()
import signal
import sys
import atexit


def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    sys.exit(0)


def alert( msj=''):
    print(msj)


def release_capture(cap):
    if cap.isOpened():
        cap.release()
        logger.info("Capture released")
    

def main(cfg, demo=True, benchmark=True, save_vid=False):

    logger.info(f'Inicializando...')

    alpr = ALPR(cfg['modelo'])
    video_path = cfg['video']['fuente']
    default_width= 640
    default_height = 480
    cap = cv2.VideoCapture(video_path, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, default_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, default_height)

    atexit.register(release_capture, cap)

    logger.info(f'Se va analizar la fuente: {video_path}')
    # Cada cuantos frames hacer inferencia
    patentes_detectadas = OrderedDict()
    max_patentes = 10
    avg = 0
    logger.debug("start loop")
    count = 0
    while cap.isOpened():
        return_value, frame = cap.read()
        count += 30
        cap.set(cv2.CAP_PROP_POS_FRAMES, count)
        if not return_value:
            continue

        #frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_w_pred, avg,patente_actual, roi = alpr.mostrar_predicts(frame)
        if patente_actual == '':
            continue 

        alert('Patente detectada : ' + patente_actual)
        # if avg < cfg['modelo']['confianza_avg_ocr']:
        #     continue

        if patente_actual in patentes_detectadas and time.time() - patentes_detectadas[patente_actual] < 120:
            print("La patente {} ya se ha procesado en los últimos 2 minutos.".format(patente_actual))
            continue
        else:
            patentes_detectadas[patente_actual] = time.time()
        
        if len(patentes_detectadas) > max_patentes:
            patentes_detectadas.popitem(last=False)  # Elimina la primera patente ingresada
    


if __name__ == '__main__':
    try:
        parser = ArgumentParser()
        parser.add_argument("--cfg", dest="cfg_file", help="Path del archivo de config, \
                            default: ./config.yaml", default='config.yaml')
        parser.add_argument("--demo", dest="demo",
                            action='store_true', help="En vez de guardar las patentes, mostrar las predicciones")
       
        parser.add_argument("--benchmark", dest="bench",
                            action='store_true', help="Medir la inferencia (incluye todo el pre/post processing")
        args = parser.parse_args()
        with open(args.cfg_file, 'r') as stream:
            try:
                cfg = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                logger.exception(exc)
        
        signal.signal(signal.SIGINT, signal_handler)
        main(cfg, args.demo, args.bench)
    except Exception as e:
        logger.exception(e)
