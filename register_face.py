import cv2
import os
import cvlib as cv

a=True
while a:
    name = input("Enter name: ")
    if name == "":
        a=True
    else:
        a=False


directory = name
parent_dir = os.getcwd()+'\Image Dataset'
path = os.path.join(parent_dir, directory)
os.mkdir(path)

vid = cv2.VideoCapture(0)
i=0
while(True):
    i+=1
    ret, frame = vid.read()
    face, confidence = cv.detect_face(frame)

    cv2.imshow('frame', frame)

    if len(face)>0 and cv2.waitKey(1) & 0xFF == ord('0') :
        cv2.imwrite('Image Dataset\\'+name+'\\1.jpg',frame) 
        return_value,image = vid.read()
    elif len(face)>0 and cv2.waitKey(1) & 0xFF == ord('1') :
        cv2.imwrite('Image Dataset\\'+name+'\\2.jpg',frame)
        return_value,image = vid.read()
    elif len(face)>0 and cv2.waitKey(1) & 0xFF == ord('2') :
        cv2.imwrite('Image Dataset\\'+name+'\\3.jpg',frame)
        return_value,image = vid.read()
    elif len(face)>0 and cv2.waitKey(1) & 0xFF == ord('3') :
        cv2.imwrite('Image Dataset\\'+name+'\\4.jpg',frame)
        return_value,image = vid.read()
    elif len(face)>0 and cv2.waitKey(1) & 0xFF == ord('4') :
        cv2.imwrite('Image Dataset\\'+name+'\\5.jpg',frame)
        return_value,image = vid.read()
    elif len(face)>0 and cv2.waitKey(1) & 0xFF == ord('5') :
        cv2.imwrite('Image Dataset\\'+name+'\\6.jpg',frame)
        return_value,image = vid.read()
    elif len(face)>0 and cv2.waitKey(1) & 0xFF == ord('6') :
        cv2.imwrite('Image Dataset\\'+name+'\\7.jpg',frame)
        return_value,image = vid.read()
    elif len(face)>0 and cv2.waitKey(1) & 0xFF == ord('7') :
        cv2.imwrite('Image Dataset\\'+name+'\\8.jpg',frame)
        return_value,image = vid.read()
    elif len(face)>0 and cv2.waitKey(1) & 0xFF == ord('8') :
        cv2.imwrite('Image Dataset\\'+name+'\\9.jpg',frame)
        return_value,image = vid.read()
    elif len(face)>0 and cv2.waitKey(1) & 0xFF == ord('9') :
        cv2.imwrite('Image Dataset\\'+name+'\\10.jpg',frame)
        return_value,image = vid.read()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()