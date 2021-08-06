import cv2
import argparse
import time

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('Downloads/haarcascade_frontalface_alt.xml')
left_eye_cascade = cv2.CascadeClassifier('Downloads/haarcascade_lefteye_2splits.xml')
right_eye_cascade = cv2.CascadeClassifier('Downloads/haarcascade_righteye_2splits.xml')
nose_cascade = cv2.CascadeClassifier('Downloads/haarcascade_smile.xml')
mouth_cascade = cv2.CascadeClassifier('Downloads/haarcascade_eye.xml')

if not cap.isOpened():
    raise IOError("Cannot open webcom")

for i in list(range(0,35)):
    print(i)
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=0.25, fy=0.25, interpolation = cv2.INTER_AREA)

    front_face = face_cascade.detectMultiScale(frame, scaleFactor = 1.1, minNeighbors = 5)

    for (x,y,w,h) in front_face:
        start = time.perf_counter()
        frame = cv2.rectangle(frame, (x,y), (x+w, y+h),(5, 67, 34), 4)
        center = (x + w//2, y + h//2)
        
        end = time.perf_counter()
        print("Face Time: {}".format(end - start))
 
        start = time.perf_counter()
        nose = nose_cascade.detectMultiScale(frame, scaleFactor = 1.1, minNeighbors = 7)
        for(x2,y2,w2,h2) in nose:
            frame = cv2.ellipse(frame, center, (w2//15, h2//15), 0, 0, 360,(45, 0, 70), 2)
        end = time.perf_counter()
        print("Nose Time: {}".format(end - start))

        start = time.perf_counter()
        mouth = mouth_cascade.detectMultiScale(frame, scaleFactor = 1.1, minNeighbors = 3)
        for(x3,y3,w3,h3) in mouth:
            mouth_center = (x + x3 + w3//2, y + y3 + h3//2)
            radius = int(round((w3 + h3)* 0.5))
            frame = cv2.circle(frame, mouth_center, radius,(123, 123, 123), 2)
            
        end = time.perf_counter()
        print("Mouse Time: {}".format(end - start))

    start = time.perf_counter()
    left_eyes = left_eye_cascade.detectMultiScale(frame, scaleFactor = 1.1, minNeighbors = 5)
    for (x2,y2,w2,h2) in left_eyes:
        frame = cv2.rectangle(frame, (x2,y2), (x2+w2, y2+h2),(255, 0, 255), 4)

    end = time.perf_counter()
    print("Left Eye Time: {}".format(end - start))


    start = time.perf_counter()
    right_eyes = right_eye_cascade.detectMultiScale(frame, scaleFactor = 1.1, minNeighbors = 5)
    for (x3,y3,w3,h3) in right_eyes:
        frame = cv2.rectangle(frame, (x3,y3), (x3+w3, y3+h3),(255, 0, 255), 4)

    end = time.perf_counter()
    print("Right Eye Time: {}".format(end - start))

    

    cv2.imshow("Input", frame)
    g = cv2.waitKey(1)

    if g == ord('g'):
        break


cap.release()
cv2.destroyAllWindows()
