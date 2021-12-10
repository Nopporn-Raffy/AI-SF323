from imutils import paths
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os
from keras.layers import Dropout
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import pickle
from sklearn.preprocessing import LabelEncoder

protoPath = "face_recog/face_detection_model/deploy.prototxt"

modelPath = "face_recog/face_detection_model/res10_300x300_ssd_iter_140000.caffemodel"

detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# load our serialized face embedding model from disk

embedder = cv2.dnn.readNetFromTorch("face_recog/openface.nn4.small2.v1.t7")

imagePaths = []
main_dir = 'Image Dataset'
main_dir_list = os.listdir(main_dir)
for name in main_dir_list:
    name_dir = os.path.sep.join([main_dir, name])
    name_dir_list = os.listdir(name_dir)
    for image in name_dir_list:
        imagePath = os.path.sep.join([main_dir, name, image])
        imagePaths.append(imagePath)

for i in imagePaths:
    print(i.split(os.path.sep)[-2], " : ",i)
    print()

# initialize our lists of extracted facial embeddings and corresponding people names
knownEmbeddings = []
knownNames = []
# initialize the total number of faces processed
total = 0

for i in range(len(imagePaths)):
    
    imagePath = imagePaths[i]
    # extract the person name from the image path
    print("[INFO] processing image {}/{}".format(i + 1, len(imagePaths)))
    name = imagePath.split(os.path.sep)[-2]
    
    # load the image, resize it to have a width of 600 pixels (while maintaining the aspect ratio), and then grab the image
    # dimensions
    
    image = cv2.imread(imagePath)
    (h, w) = image.shape[:2]
    
    # construct a blob from the image
    
    imageBlob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300),(104.0, 177.0, 123.0),
                                      swapRB=False, crop=False)
    
    # apply OpenCV's deep learning-based face detector to localize faces in the input image
    
    detector.setInput(imageBlob)
    detections = detector.forward()
    
    if len(detections) > 0:
        
        # we're making the assumption that each image has only ONE face, so find the bounding box with the largest probability
        
        i = np.argmax(detections[0, 0, :, 2])
        confidence = detections[0, 0, i, 2]
        
        # ensure that the detection with the largest probability also
        # means our minimum probability test (thus helping filter out
        # weak detections)
        
        if confidence > 0.5:
        
            # compute the (x, y)-coordinates of the bounding box for the face
        
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            
            # extract the face ROI and grab the ROI dimensions
            
            face = image[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]
            
            # ensure the face width and height are sufficiently large
            
            if fW < 20 or fH < 20:
                continue
            
            # construct a blob for the face ROI, then pass the blob through our face embedding model to obtain the 128-d
            # quantification of the face
            
            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
            embedder.setInput(faceBlob)
            vec = embedder.forward()
            
            # add the name of the person + corresponding face
            # embedding to their respective lists
            
            knownNames.append(name)
            knownEmbeddings.append(vec.flatten())
            total += 1

data = {"embeddings": knownEmbeddings, "names": knownNames}
f = open('Trined_Embeddings', "wb")
f.write(pickle.dumps(data))
f.close()

model = Sequential([Dense(128, input_shape = (128,), activation = 'relu'),
                    Dense(64, activation = 'relu', kernel_initializer = 'he_uniform'),
                    Dense(64, activation = 'relu',  kernel_initializer = 'he_uniform'),
                    Dense(32, activation = 'relu',  kernel_initializer = 'he_uniform'),
                    Dense(32, activation = 'relu',  kernel_initializer = 'he_uniform'),
                    Dense(5, activation = 'softmax')])

model.compile(loss='sparse_categorical_crossentropy', optimizer = 'adam' ,metrics=['accuracy'])
model.summary()

data = pickle.loads(open('Trined_Embeddings', "rb").read())

le = LabelEncoder()
labels = le.fit_transform(data["names"])


embeddings = data["embeddings"]
embeddings = np.array(embeddings)
h = model.fit(embeddings, labels, epochs = 50)

# write the actual face recognition model to disk
model.save('recognizer.h5')
# write the label encoder to disk
f = open("label_Encoder", "wb")
f.write(pickle.dumps(le))
f.close()