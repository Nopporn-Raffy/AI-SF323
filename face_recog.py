import numpy as np
import pickle
import cv2
import os
import pickle
from keras.models import load_model
import face_detection as fd

protoPath = "face_recog/face_detection_model/deploy.prototxt"

modelPath = "face_recog/face_detection_model/res10_300x300_ssd_iter_140000.caffemodel"

detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

embedder = cv2.dnn.readNetFromTorch("face_recog/openface.nn4.small2.v1.t7")

recognizer = load_model('recognizer.h5')

le = pickle.loads(open("label_Encoder", "rb").read())

def image_recognize(image):
    image = cv2.resize(image, (640, 480))
    (h, w) = image.shape[:2]
    imageBlob = cv2.dnn.blobFromImage( image, 1.0, (300, 300), (104.0, 177.0, 123.0),
                                      swapRB=False, crop=False)
    
    detector.setInput(imageBlob)
    detections = detector.forward()
    
    for i in range(0, detections.shape[2]):
    
        # extract the confidence (i.e., probability) associated with the prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections
        if confidence > 0.85:
            # compute the (x, y)-coordinates of the bounding box for the face

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # extract the face ROI

            face = image[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]

            # ensure the face width and height are sufficiently large
    
            if fW < 30 or fH < 30:
                continue
                
            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
            embedder.setInput(faceBlob)
            vec = embedder.forward()

            # perform classification to recognize the face
            
            preds = recognizer.predict(vec)[0]
            j = np.argmax(preds)

            proba = preds[j]
            name = le.classes_[j]
            if proba >= 0.90:
                text = name

                cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
                cv2.putText(image, text, (startX, startY), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 0), 2)
            else :
                text = "Unknow"

                cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
                cv2.putText(image, text, (startX, startY), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 0), 2)

    return np.array(image)

def main() :
    imagePaths = []
    main_dir = 'test_images'
    main_dir_list = os.listdir(main_dir)
    for image in main_dir_list:
        imagePath = os.path.sep.join([main_dir, image])
        imagePaths.append(imagePath)

    image = cv2.imread(imagePaths[0])
    image = image_recognize(image)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ret, buffer = cv2.imencode('.jpg', image)
    f = buffer.tobytes()
    yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + f + b'\r\n')