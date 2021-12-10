from face_recognition.api import face_landmarks
from numpy.lib.type_check import imag
import numpy as np
import cv2
import os
import cvlib as cv
import threading
import face_recognition

list_encode = []
webcam = cv2.VideoCapture(0)
width = int(webcam.get(3))
height = int(webcam.get(4))

def encodeImg() :
    try:
        image = face_recognition.load_image_file("./image/face.jpg")
        image_encodings = face_recognition.face_encodings(image)[0]

        card_image = face_recognition.load_image_file("./image/card.jpg")
        card_encodings = face_recognition.face_encodings(card_image)[0]

        list_encode.append([image_encodings, card_encodings])

        print("encoded!!!")

    except :
        pass

def detectFace(face, frame) :
    card = False
    img = False
    try :
        for idx, f in enumerate(face): 
            (startX, startY) = f[0], f[1]
            (endX, endY) = f[2], f[3]

            if endX <= int(width/2)-25 and endY <= int(height/2)+75 and startX > 25 and startY > 150:
                cv2.imwrite("image/card.jpg",frame[startY:endY,startX:endX])
                cv2.rectangle(frame, (startX,startY), (endX,endY), (0,255,0), 1)
                card = True 

            if endX >= width/2 :
                cv2.imwrite("image/face.jpg",frame[startY:endY,startX:endX])
                cv2.rectangle(frame, (startX,startY), (endX,endY), (0,255,0), 1)
                img = True

            cv2.imwrite("test_images/full.jpg", frame)
    except :
        pass

    if card and img :
        return True
    else :
        return False

def main() :
    try:
        os.remove("image/face.jpg")
        os.remove("image/card.jpg")
    except:
        print("no file to remove")

    finally :
        # loop through frames
        while True:# while webcam.isOpened():
            # read frame from webcam 
            status, frame = webcam.read()

            # apply face detection
            face, confidence = cv.detect_face(frame)
            cv2.rectangle(frame,(25, 150), (int(width/2)-25, int(height/2)+75), (0, 255, 0), 2)

            saved = detectFace(face, frame)
            # cv2.imshow("detection", frame)
            ret, buffer = cv2.imencode('.jpg', frame)
            f = buffer.tobytes()
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + f + b'\r\n')

            if len(list_encode) < 1 :
                if saved :
                    encodeThread = threading.Thread(target=encodeImg())
                    encodeThread.start()

                # press "Q" to stop
                if cv2.waitKey(1) & 0xFF == ord('q') :
                    break
            else :
                break
            
        # release resources
        try:
            face_distances = face_recognition.face_distance([list_encode[0][0]], list_encode[0][1])
            result = face_recognition.compare_faces([list_encode[0][0]], list_encode[0][1])
            # list_distance.append(face_distances)
            # list_compare.append(result)
            # percentage = (1 - face_distances[0])*100
            percentage = (1 - face_distances[0])*100
            percentage = "{:.2f}".format(percentage)

            cv2.putText(frame, ("Percent: " + percentage +" Match: "+ str(result[0])),(10,50),cv2.FONT_HERSHEY_SIMPLEX,1,(209, 80, 0, 255),2)
            # print("Percentage: {percentage}     Math: {result}")
            ret, buffer = cv2.imencode('.jpg', frame)
            f = buffer.tobytes()
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + f + b'\r\n')
        except Exception as e :
            print("Error: ",e)

        webcam.release()
        cv2.destroyAllWindows()