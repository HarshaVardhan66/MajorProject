from django.core.files.storage import FileSystemStorage
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render, redirect
from django.http import HttpResponse
from twilio.rest import Client
from .models import Susinfo
from PIL import Image
import numpy as np
import cv2
import os


def homepage(request):
    return render(request, 'FaceDetect/home.html')


def services(request):
    return render(request, 'FaceDetect/services.html')


def howto(request):
    return render(request, 'FaceDetect/howto.html')


def upload(request):
    return render(request, 'FaceDetect/upload.html')

@csrf_exempt
def live_capture(request):
    data = request.POST
    cap = cv2.VideoCapture(0)
    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.imshow('capturing', frame)
            cv2.imwrite('E:\\MajorProject\\FaceDetect\\Faces\\' + data['sus_id'] + '.' + str(i) + '.jpg', frame)
            i += 1
            if i == 50 or cv2.waitKey(1) == 27:
                break
    cap.release()
    cv2.destroyAllWindows()

    suspect = Susinfo()
    suspect.name = data['sus_name']
    suspect.susid = data['sus_id']
    suspect.auth = data['auth_name']
    suspect.phone = data['auth_phone']
    suspect.isSuspect = data['check']
    suspect.save()

    f = open("E:\\MajorProject\\trainingdetails.txt", "r")
    s = f.read()
    f.close()

    l = [x for x in s.split('.')]
    s = l[0] + '.' + str(int(l[1]) + 1)

    f = open("E:\\MajorProject\\trainingdetails.txt", "w")
    f.write(s)
    f.close()

    di = {'trained': int(l[0]), 'remaining': int(l[1])+1}

    return render(request, 'FaceDetect/uploadresult.html', context=di)


@csrf_exempt
def video_submit(request):

    data = request.POST
    filename = data['sus_id']+'.mp4'
    video_file = request.FILES['video']

    fs = FileSystemStorage()
    fs.save(filename, video_file)
    cap = cv2.VideoCapture('E:\\MajorProject\\media\\'+filename)
    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.imshow('capturing', frame)
            cv2.imwrite('E:\\MajorProject\\FaceDetect\\Faces\\' + data['sus_id'] + '.' + str(i) + '.jpg', frame)
            i += 1
            if i == 50 or cv2.waitKey(1) == 27:
                break
    cap.release()
    cv2.destroyAllWindows()
    suspect = Susinfo()
    suspect.name = data['sus_name']
    suspect.susid = data['sus_id']
    suspect.auth = data['auth_name']
    suspect.phone = data['auth_phone']
    suspect.isSuspect = data['check']
    suspect.save()

    f = open("E:\\MajorProject\\trainingdetails.txt", "r")
    s = f.read()
    f.close()

    l = [x for x in s.split('.')]
    s = l[0] + '.' + str(int(l[1]) + 1)

    f = open("E:\\MajorProject\\trainingdetails.txt", "w")
    f.write(s)
    f.close()

    di = {'trained': int(l[0]), 'remaining': int(l[1]) + 1}
    return render(request, 'FaceDetect/uploadresult.html', context=di)


def train(request):
    f = open("E:\\MajorProject\\trainingdetails.txt", "r")
    s = f.read()
    f.close()
    li = [x for x in s.split('.')]
    di = {'trained': int(li[0]), 'remaining': int(li[1])}
    return render(request, 'FaceDetect/train.html', context=di)


def trainingresult(request):
    path = 'E:\\MajorProject\\FaceDetect\\Faces'
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier("E:\\MajorProject\\FaceDetect\\Cascades\\haarcascade_frontalface_default.xml")

    def getImagesAndLabels(path1):
        imagePaths = [os.path.join(path1, f) for f in os.listdir(path1)]

        faceSamples = []
        ids1 = []

        for imagePath in imagePaths:
            print(imagePath)
            PIL_img = Image.open(imagePath).convert('L')  # grayscale
            img_numpy = np.array(PIL_img, 'uint8')
            id1 = int(os.path.split(imagePath)[-1].split(".")[0])
            faces = detector.detectMultiScale(img_numpy)

            for (x, y, w, h) in faces:
                faceSamples.append(img_numpy[y:y + h, x:x + w])
                ids1.append(id1)

        f = open("E:\\MajorProject\\trainingdetails.txt", "r")
        s = f.read()
        print(s)
        f.close()
        l = [x for x in s.split('.')]
        s = str(int(l[1])+int(l[0])) + '.' + str(0)
        f = open("E:\\MajorProject\\trainingdetails.txt", "w")
        print(s)
        f.write(s)
        f.close()
        return faceSamples, ids1

    faces1, ids = getImagesAndLabels(path)
    recognizer.train(faces1, np.array(ids))

    recognizer.write('trainer.yml')
    a = len(np.unique(ids))
    di = {'trained': a}
    return render(request, 'FaceDetect/trainingresult.html', context=di)

def recognise(request):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # out= cv2.VideoWriter('output.mp4', fourcc, 30, (640,480))
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('E:\\MajorProject\\trainer.yml')
    cascadePath = "E:\\MajorProject\\FaceDetect\\Cascades\\haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascadePath)
    font = cv2.FONT_HERSHEY_SIMPLEX

    # names related to ids: example ==> Marcelo: id=1,  etc
    names = {0 :'None'}
    sus_ids= []
    for i in Susinfo.objects.all():
        names[int(i.susid)] = i.name
        if i.isSuspect == 'yes':
            sus_ids.append(int(i.susid))
    print(names, sus_ids)

    # Initialize and start realtime video capture
    cam = cv2.VideoCapture(0)
    cam.set(3, 640)  # set video widhth
    cam.set(4, 480)  # set video height
    # Define min window size to be recognized as a face
    minW = 0.1 * cam.get(3)
    minH = 0.1 * cam.get(4)

    sus_detected = []
    non_sus_detected = []
    while True:
        ret, img = cam.read()
        img = cv2.flip(img, 1)  # Flip vertically
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(int(minW), int(minH)))
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

            # If confidence is less them 100 ==> "0" : perfect match
            if confidence < 80:
                disp_name = names[id]
            else:
                disp_name = "unknown"

            if id in sus_ids:
                if id not in sus_detected:
                    acc_sid = "AC361b640ba71f76a616d917cd84e6fbac"
                    auth = "f0e76d88ce5edbb555c23858a93f2f76"
                    client = Client(acc_sid, auth)
                    client.messages.create(from_="+18329063590", body="!!! The "+names[id]+"-"+ str(id) + " is spotted here !!!", to="+919866530705")

                    sus_detected.append(id)
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            else:
                if id not in non_sus_detected:
                    non_sus_detected.append(id)
            cv2.putText(img, str(disp_name), (x + 5, y - 5), font, 1, (255, 255, 255), 2)

        cv2.imshow('camera', img)
        # out.write(img)
        k = cv2.waitKey(10) & 0xff  # Press 'ESC' for exiting video
        if k == 27:
            break
    sus_details = {}
    non_sus_details = {}
    for i in sus_detected:
        sus_details[i] = names[i]
    for i in non_sus_detected:
        non_sus_details[i] = names[i]
    cam.release()
    cv2.destroyAllWindows()

    di = {'sus_detected':sus_details, 'non_sus_detected':non_sus_details, 'tot': len(sus_detected)+len(non_sus_detected), 'sus':len(sus_detected), 'non_sus':len(non_sus_detected)}
    return render(request, 'FaceDetect/recogniseresult.html', context=di)