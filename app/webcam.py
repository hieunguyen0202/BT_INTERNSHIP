import threading
from file_main import User
from flask import Flask, render_template,Response,redirect, url_for, request, session, flash
import cv2
from app.file_main import app
from keras.models import load_model
from keras.preprocessing import image
import keras
import face_recognition
from datetime import datetime
import numpy as np
from sqlalchemy import Column, Integer, String, Boolean, ForeignKey, Float, DateTime, Boolean, Enum
from keras.applications.mobilenet_v2 import preprocess_input
from keras.utils import img_to_array
from imutils.video import VideoStream
import imutils
import time
from app.file_main import db,user
import os

import csv
import json

from threading import Thread
# load our serialized face detector model from disk
prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
maskNet = load_model("mask_detector.model")
# model = load_model("model_train_da_xong.h5")

text_dis = []
camera = cv2.VideoCapture(0)
camera1 = cv2.VideoCapture("include_image/cam2.gif")
# face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
# recoginite_name = session['name']
# nhan dien nguoi
path = 'ImagesBasic'
images = []
classNames = []
myList = os.listdir(path)
print(myList)



for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

print(classNames)

# def convert():
#  while True:
#     with open("atten.csv", "r+") as f:
#         reader = csv.reader(f)
#         data = []
#
#         for row in reader:
#             if row[0] and row[1] == "":
#                 data.append({'Name': 'none', 'Time': 'none'})
#
#             else:
#                 data.append({'Name': row[0], 'Time': row[1]})
#
#
#     with open("static/images/new02.json", "w+") as f:
#         json.dump(data, f, indent=4)
#
# #
# #
# threadso1 = threading.Thread(target=convert)
# threadso1.start()





# def convert():
#     with open("atten.csv", "r") as f:
#         reader = csv.reader(f)
#         data = []
#         for row in reader:
#             data.append({'Name': row[0], 'Time': row[1]})
#
#     with open("new.json", "w") as f:
#         json.dump(data, f, indent=4)

def findEncodings(image):
    encodeList = []
    for img in image:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList
#
# def markAttendance(name):
#     with open('atten.csv', 'r+') as f:
#         myDataList = f.readlines()
#         # print(myDataList)
#         nameList = []
#         for line in myDataList:
#             entry = line.split(',')
#             nameList.append(entry[0])
#         if name not in nameList:
#             now = datetime.now()
#             dtString = now.strftime('%H:%M:%S')
#             f.writelines(f'\n{name},{dtString}')

def ktraJson():
    with open ('static/images/new02.json', 'r') as f:
        temp = json.load(f)
        for entry in temp:
            name = entry["Name"]
            stt = entry["STT"]

    return name,stt


def addfileJson(name):
    now = datetime.now()
    dtString = now.strftime('%H:%M:%S')
    item_data = {}
    with open ('static/images/new02.json', 'r') as f:
        temp = json.load(f)
        for entry in temp:
            pre_name = entry["Name"]
            stt = entry["STT"]

    new_name = name
    ktNewName = User.query.filter_by(email=new_name).first()

    if name != pre_name:
        with open ('static/images/new02.json', 'r') as f:
            temp = json.load(f)
        item_data["STT"] = stt + 1
        item_data["Name"] = name
        item_data["Time"] = dtString
        if ktNewName:
            item_data["UserName"] = ktNewName.username
            item_data["Department"] = ktNewName.department
            item_data["DateBirth"] = ktNewName.date_birth
        else:
            item_data["UserName"] = "none"
            item_data["Department"] = "none"
            item_data["DateBirth"] = "none"
        temp.append(item_data)


        with open ('static/images/new02.json', 'w') as f:
            json.dump(temp, f, indent=4)



# nhan dien nguoi


encodeListKnown = findEncodings(images)

stop_thread = threading.Event()

def offcam():
    # stop_thread.set()
    while True:
        ret, cam = camera1.read()
        if not ret:
            break
        else:
            ret, buffer1 = cv2.imencode('.jpg', cam)
            cam = buffer1.tobytes()

        yield (b'--cam\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + cam + b'\r\n')






def generate_frames():
    while True:
        success,frame = camera.read()
        # nhan dien nguoi
        cv2.imwrite('include_image/cam3.jpg', frame)
        # nhan dien nguoi
        # khau trang
        frame = imutils.resize(frame, width=400)
        (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)
        for (box, pred) in zip(locs, preds):
            # unpack the bounding box and predictions
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred
            # determine the class label and color we'll use to draw
            # the bounding box and text
            label = "Mask" if mask > withoutMask else "No Mask"
            face_img = frame[startY + 3: endY - 3, startX + 3: endX - 3]
            image_faces_size = cv2.resize(frame[startY + 3: endY - 3, startX + 3: endX - 3], (100, 120))
            # cv2.imwrite('static/images/Anh_{}.jpg'.format(count), image_faces_size)

            # color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
            count = 0
            count1 = 2
            if label == "Mask":

                color = (0, 255, 0)
                cv2.imwrite('static/images/Anh_{}.jpg'.format(count), image_faces_size)
                count += 1
                # cv2.imwrite('static/images/Anh_1.jpg'.format(count), image_faces_size)
            else:
                color = (0,0,255)
                cv2.imwrite('static/images/Anh_{}.jpg'.format(count1), image_faces_size)
                count1 += 1
                # cv2.imwrite('static/images/Anh_3.jpg'.format(count1), image_faces_size)
			# include the probability in the label
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            cv2.putText(frame, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
            count = 0
            # khau trang


        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        # if stop_thread.is_set():
        #     break

def nhan_dang():
    i = 0
    while True:
            time.sleep(5)
            frame = cv2.imread('include_image/cam3.jpg')
            imgS = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
            imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
            facesCurFrame = face_recognition.face_locations(imgS)
            encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)
            for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
                matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
                faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
                # print(faceDis)
                matchIndex = np.argmin(faceDis)
                if matches[matchIndex]:
                    name = classNames[matchIndex].upper()
                    # print(name)
                    y1, x2, y2, x1 = faceLoc
                    y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                    cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

                    addfileJson(name)



thread10 = Thread(target=nhan_dang)
thread10.start()

def detect_and_predict_mask(frame, faceNet, maskNet):
	# grab the dimensions of the frame and then construct a blob
	# from it
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
		(104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the face detections
	faceNet.setInput(blob)
	detections = faceNet.forward()
	print(detections.shape)

	# initialize our list of faces, their corresponding locations,
	# and the list of predictions from our face mask network
	faces = []
	locs = []
	preds = []

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the detection
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		if confidence > 0.5:
			# compute the (x, y)-coordinates of the bounding box for
			# the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of
			# the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			# add the face and bounding boxes to their respective
			# lists
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# only make a predictions if at least one face was detected
	if len(faces) > 0:
		# for faster inference we'll make batch predictions on *all*
		# faces at the same time rather than one-by-one predictions
		# in the above `for` loop
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	# return a 2-tuple of the face locations and their corresponding
	# locations
	return (locs, preds)



@app.route('/hoa')
def index():

    return render_template('layout/user.html')

@app.route('/video')
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/offvideo')
def offvideo():
    return Response(offcam(),mimetype='multipart/x-mixed-replace; boundary=cam')

if __name__ == "__main__":
    app.run(debug=True)


cv2.destroyAllWindows()

# face = face_detector.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=4)
# count = 0
# for (x, y, w, h) in face:
#     image_faces_size = cv2.resize(frame[y + 3: y + h - 3, x + 3: x + w - 3], (100, 120))
#     face_img = frame[y: y + h, x: x + w]
#     cv2.imwrite('static/images/Anh_{}.jpg'.format(count), image_faces_size)
#     cv2.imwrite('Hienthi.jpg', face_img)
#     # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
#     count += 1
#
#
#     test_image = tf.keras.utils.load_img('Hienthi.jpg', target_size=(150, 150, 3))
#     test_image = tf.keras.utils.img_to_array(test_image)
#     test_image = np.expand_dims(test_image, axis=0)
#     predict_image = model.predict(test_image)[0][0]
#     # # Test_image = predict_image
#
#     if predict_image == 1:
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
#         cv2.putText(frame, 'KHONG DEO KHAU TRANG', ((x + w) // 2, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 1,
#                     (0, 1, 25), 3)
#         # return render_template('layout/user.html',megs="warning: no mask")
#     else:
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
#         cv2.putText(frame, 'CO DEO KHAU TRANG', ((x + w) // 2, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 1,
#                     (0, 1, 25), 3)
#         # return render_template('layout/user.html',megs="normal")
#
# if cv2.waitKey(1) & 0xFF == ord('q'):
#     break