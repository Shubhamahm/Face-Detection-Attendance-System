import cv2
import numpy as np
import face_recognition
import os
import pandas as pd
from datetime import datetime

# Load known faces
path = 'Images'  # Folder containing images of known faces
images = []
classNames = []
myList = os.listdir(path)

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])  # Extract name without extension

# Encode known faces
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)
        if encode:
            encodeList.append(encode[0])
    return encodeList

encodeListKnown = findEncodings(images)
print("Encoding Complete")

# Function to mark attendance
def markAttendance(name):
    file = 'Attendance.xlsx'
    now = datetime.now()
    dtString = now.strftime('%Y-%m-%d %H:%M:%S')
    
    if not os.path.exists(file):
        df = pd.DataFrame(columns=['Name', 'Date Time'])
        df.to_excel(file, index=False)
    
    df = pd.read_excel(file)
    if name not in df['Name'].values:
        new_data = pd.DataFrame({'Name': [name], 'Date Time': [dtString]})
        df = pd.concat([df, new_data], ignore_index=True)
        df.to_excel(file, index=False)

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    
    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)
    
    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)
        
        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            markAttendance(name)
            
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()