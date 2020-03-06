# Text Detection 

This project aimed at extracting texts from given Images. Text extraction consists of two parts. First is detecting the text area and then scanning the area to recognise the texts present there.

## Instructions on How To Use
Two methods has been used to detect texts from the image and then the detected area is then fed to pytessaract to recognise the text.

1.) For using EAST run 
'''  python text_recognition.py --east frozen_east_text_detection.pb --folder DATASET '''

2.) For using keras OCR
''' python keras_ocr_test.py --dataset PATH_TO_DATASET '''

## You Must Have Following Libraries Installed
sklearn
matplotlib
opencv
keras_ocr
pytessaract
argparse


