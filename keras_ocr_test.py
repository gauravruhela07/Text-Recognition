import keras_ocr
import pytesseract
import os, cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse

from extract_poly import extract_roi
from text_preprocessing import cleanString

detector = keras_ocr.detection.Detector()

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--dataset", type=str, help = "path to dataset")
args = vars(ap.parse_args())
dataset = args['dataset']

# Boxes will be an Nx4x2 array of box quadrangles
# where N is the number of detected text boxes.


# print(boxes)

image_dataset = dataset
# result = 'result'
text_file = open('images_to_text_py_ocr.csv','w')
text_file.write('py_ocr,texts\n')
cnt = 0
for f in tqdm(os.listdir(image_dataset)):
    img = cv2.imread(os.path.join(image_dataset,f))

    image = keras_ocr.tools.read(os.path.join(image_dataset, f))
    boxes = detector.detect(images=[image])[0]

    text_file.write(f+',')
    texts = []
    for i in range(len(boxes)):
        origW, origH = img.shape[0], img.shape[1]
        # roi = img[40:184, 153:219]

        
        # print(startX+1, startY, endX-1, endY-1)

        roi2 = extract_roi(img, boxes[i])
        # roi2 = img[startY:endY, startX:endX]
        # print(roi.shape)
        # cv2.im
        config = ("-l eng --oem 1 --psm 7")
        text = pytesseract.image_to_string(roi2, config=config)
        text = cleanString(text)
        texts.append(text)
    st = ""
    for t in texts:
        st+= t+ ' '
    text_file.write(st+'\n')
    cnt+=1
text_file.close()
        # cv2.destroyAllWindows()
        # plt.imshow(img)
        # plt.imshow(roi2)
        # plt.show()
