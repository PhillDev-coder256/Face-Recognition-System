# Project Name : Face Recognition System
# Author : Ssempeebwa Phillip

# Importing necessary libraries
import cv2 as cv
import numpy as np
import os
import sqlite3
import time
from datetime import *
from PIL import Image

# *********************************Collect the data to train the detector************************
                                    # Call this function once

def collect_data():
    detector = cv.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
    cap = cv.VideoCapture(0)
    cap.set(3,640) #set the width
    cap.set(4,480) #set the height
    
    # collect the data
    face_id = input('\n Enter Your Id: ')
    student_name = input('\n Enter your name: ')
    student_gender = input('\n Enter your gender: ')

    student_details = (face_id, student_name, student_gender)

    # connect the database
    conn = sqlite3.connect('students.db')
    cur = conn.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS students (id,name,gender)")

    cur.execute("INSERT INTO students VALUES(?,?,?)", student_details)
    conn.commit()

    print("\n Initializing face capture. Look at the camera and wait ...")
    count = 0

    while True:
        ret, frame = cap.read()
        gray = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        faces = detector.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(20,20)
        )

        for (x,y,w,h) in faces:
            cv.rectangle(frame, (x,y), (x+y, y+h), (255,0,0),2)
    #     gray = cv.cvtColor(frame, cv.COLOR_BGR2RGB
            count += 1

            # Save the captured image
            cv.imwrite("dataset/User." + str(face_id) + '.' +  
                        str(count) + ".jpg", gray[y:y+h,x:x+w])
            
            # roi_gray = gray[y:y+h, x:x+w]
            # roi_color = frame[y:y+h, x:x+w]
        cv.imshow('video', frame)
        k = cv.waitKey(30) & 0xFF
        if k == 27:
            break
        elif count >= 200: #collect 30 photos and close the video
            break
    # cleaning up stuff     
    cap.release()
    cv.destroyAllWindows()

    return f'Data collection successful, next train your recorgniser with the collected data.'

 
 # *********************************train the recorginer with our dataset************************
                                    # Call this function once


def train_recorgniser():
    path = "dataset"
    detector = cv.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')

    recorgniser = cv.face.LBPHFaceRecognizer_create()
    # recorgniser = cv.
    # get image data and respective labels
    def getImageData(path):
        images = [os.path.join(path,f) for f in os.listdir(path)]
        samples = []
        ids = []

        for imge in images:
            pillow_image = Image.open(imge).convert('L') #Grayscale image for opencv2
            # pillow_image = Image.open
            numpy_image = np.array(pillow_image, 'uint8')
            id = int(os.path.split(imge)[-1].split(".")[1])
            det_faces = detector.detectMultiScale(numpy_image)

            for (x,y,w,h) in det_faces:
                samples.append(numpy_image[y:y+h, x:x+w])
                ids.append(id)

        return samples,ids

    print('Training recorgniser on the input faces....')
    all_faces,ids = getImageData(path)
    recorgniser.train(all_faces, np.array(ids))
    # Save the model into trainer/trainer.yml
    recorgniser.write('trainer/trainer.yml') 
    print(f'Trained {0}% faces, exiting the program....'.format(len(np.unique(ids))))

    return f'Training completed, Its time to test your crappy recorgniser...'





 # *********************************test the recorginer with our dataset************************
                                    # Call this function to test the recorgniser
def test_recorgniser():
    detector = cv.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
    recorgniser = cv.face.LBPHFaceRecognizer_create()

    recorgniser.read('trainer/trainer.yml')
    font = cv.FONT_HERSHEY_SIMPLEX
    id = 0

    conn = sqlite3.connect('students.db')
    students_names = ['None',]

    cur = conn.cursor()

    students = cur.execute("SELECT * FROM students")
    print(students)
    for student in students:
        students_names.append(student[1])
        print(student)


    # initialise start realtime video capture\
    cap = cv.VideoCapture(0)
    cap.set(3,640) #set the width
    cap.set(3,640) #set the height
    minWin = 0.1*cap.get(3) #Minimum window capture to be recorgnised as a face
    minHei = 0.1*cap.get(3) #Minimum window capture to be recorgnised as a face

    while True:
        ret, img = cap.read()
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        faces = detector.detectMultiScale(
            gray,
            scaleFactor = 1.2,
            minNeighbors = 5,
            minSize = (int(minWin), int(minHei)),
        )

        for (x,y,h,w) in faces:
            cv.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 4)
            id, confidence = recorgniser.predict(gray[y:y+h,x:x+w])

            if confidence < 100 > 40:
                id = students_names[id]
                confidence = "{0}%".format(round(100 - confidence))
            else:
                id = "Unknown"
                confidence = " {0}%".format(round(100 - confidence))

            cv.putText(
                img,
                str(id),
                (x+5, y-5),
                font,
                1,
                (255, 255, 255),
                2
                    )
            cv.putText(
                img,
                str(confidence),
                (x-60, y-5),
                
                font,
                1,
                (255, 255, 0),
                1
                    )
            
        cv.imshow('camera', img)
        k = cv.waitKey(10) & 0xFF
        
        if k == 27:
            break
    print("\n Thank you for using this nerdy program, closing window and the exiting program ...")
    cap.release()
    cv.destroyAllWindows()
    return f'You successfuly built your simple face recorgnition program, Next time advance...You are great!!'



#***************************** function calls here

collect_data()
# train_recorgniser()
# test_recorgniser()


