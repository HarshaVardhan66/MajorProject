import os
import sys
import cv2
import joblib
from numpy import asarray
from .models import Susinfo
from sklearn.svm import SVC
from numpy import expand_dims
from twilio.rest import Client
from django.shortcuts import render
from django.core.cache import cache
from keras.models import load_model
from django.http import HttpResponse
from sklearn.preprocessing import Normalizer
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import FileSystemStorage

g_path = sys.path[0]
g_path = g_path.replace('\\', '/')


def homepage(request):
    cache.clear()
    return render(request, 'FaceDetect/home.html')


@csrf_exempt
def services(request):
    return render(request, 'FaceDetect/services.html')


@csrf_exempt
def howto(request):
    return render(request, 'FaceDetect/howto.html')


@csrf_exempt
def upload(request):
    return render(request, 'FaceDetect/upload.html')


@csrf_exempt
def live_capture(request):
    data = request.POST
    cap = cv2.VideoCapture(0)
    path = g_path + '/FaceDetect/Cascades/haarcascade_frontalface_default.xml'
    faceCascade = cv2.CascadeClassifier(path)
    i = 1
    path = g_path + '/FaceDetect/Faces/'
    while cap.isOpened():
        ret, img = cap.read()
        if ret:
            flag = 0
            img = cv2.flip(img, 1)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(20, 20))

            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                face = img[y:y + h, x:x + w]
                flag = 1

            cv2.imshow('video', img)
            if flag == 1:
                cv2.imwrite(path + str(data['sus_id']) + '.' + str(i) + '.jpg',
                            cv2.resize(face, (160, 160)))
                i += 1
            if i > 50 or cv2.waitKey(10) == 27 & 0xff:
                break
        else:
            break

    print("DONE")
    cap.release()
    cv2.destroyAllWindows()

    suspect = Susinfo()
    suspect.name = data['sus_name']
    suspect.susid = data['sus_id']
    suspect.auth = data['auth_name']
    suspect.phone = data['auth_phone']
    suspect.isSuspect = data['check']
    suspect.save()

    path = g_path + '/trainingdetails.txt'
    f = open(path, "r")
    s = f.read()
    f.close()

    l = [x for x in s.split('.')]
    s = l[0] + '.' + str(int(l[1]) + 1)

    f = open(path, "w")
    f.write(s)
    f.close()

    di = {'trained': int(l[0]), 'remaining': int(l[1]) + 1}

    return render(request, 'FaceDetect/uploadresult.html', context=di)


@csrf_exempt
def video_submit(request):
    data = request.POST
    filename = data['sus_id'] + '.mp4'
    video_file = request.FILES['video']

    fs = FileSystemStorage()
    fs.save(filename, video_file)

    cap = cv2.VideoCapture(g_path + '/media/' + filename)
    path = g_path + '/FaceDetect/Cascades/haarcascade_frontalface_default.xml'
    faceCascade = cv2.CascadeClassifier(path)
    i = 1
    path = g_path + '/FaceDetect/Faces/'
    while cap.isOpened():
        ret, img = cap.read()
        if ret:
            flag = 0
            img = cv2.flip(img, 1)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(20, 20))

            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                face = img[y:y + h, x:x + w]
                flag = 1

            cv2.imshow('video', img)
            if flag == 1:
                cv2.imwrite(path + str(data['sus_id']) + '.' + str(i) + '.jpg',
                            cv2.resize(face, (160, 160)))
                i += 1
            if i > 50 or cv2.waitKey(10) == 27 & 0xff:
                break
        else:
            break

    print("DONE")
    cap.release()
    cv2.destroyAllWindows()
    suspect = Susinfo()
    suspect.name = data['sus_name']
    suspect.susid = data['sus_id']
    suspect.auth = data['auth_name']
    suspect.phone = data['auth_phone']
    suspect.isSuspect = data['check']
    suspect.save()
    path = g_path + '/trainingdetails.txt'
    f = open(path, "r")
    s = f.read()
    f.close()

    l = [x for x in s.split('.')]
    s = l[0] + '.' + str(int(l[1]) + 1)

    f = open(path, "w")
    f.write(s)
    f.close()

    di = {'trained': int(l[0]), 'remaining': int(l[1]) + 1}
    return render(request, 'FaceDetect/uploadresult.html', context=di)


@csrf_exempt
def train(request):
    path = g_path + '/trainingdetails.txt'
    f = open(path, "r")
    s = f.read()
    f.close()
    li = [x for x in s.split('.')]
    di = {'trained': int(li[0]), 'remaining': int(li[1])}
    return render(request, 'FaceDetect/train.html', context=di)


@csrf_exempt
def trainingresult(request):
    def get_embedding(model, face_pixels):
        # scale pixel values
        face_pixels = face_pixels.astype('float32')
        # standardize pixel values across channels (global)
        mean, std = face_pixels.mean(), face_pixels.std()
        face_pixels = (face_pixels - mean) / std
        # transform face into one sample
        samples = expand_dims(face_pixels, axis=0)
        # make prediction to get embedding
        yhat = model.predict(samples)
        return yhat[0]

    images = []
    trainy = []
    path = g_path + '/FaceDetect/Faces/'
    for i_path in os.listdir(path):
        img = cv2.imread(path + i_path)
        images.append(img)
        trainy.append(int(os.path.split(i_path)[-1].split(".")[0]))
    images = asarray(images)
    model = load_model(g_path + '/FaceDetect/facenet_keras.h5')
    newimages = []

    for face in images:
        embeding = get_embedding(model, face)
        newimages.append(embeding)

    newimages = asarray(newimages)
    print('\n\nEmbeddings Done\n\n')
    in_encoder = Normalizer(norm='l2')
    trainX = in_encoder.transform(newimages)

    model = SVC(kernel='linear', probability=True)
    model.fit(trainX, trainy)
    joblib.dump(model, 'Recognizer.sav')
    path = g_path + '/trainingdetails.txt'
    f = open(path, "r")
    s = f.read()
    f.close()
    l = [x for x in s.split('.')]
    a = int(l[0]) + int(l[1])
    s = str(int(l[1]) + int(l[0])) + '.' + str(0)
    f = open(path, "w")
    f.write(s)
    f.close()

    di = {'trained': a}
    print('Training Done')
    return render(request, 'FaceDetect/trainingresult.html', context=di)


@csrf_exempt
def recognise(request):
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # out= cv2.VideoWriter('output.mp4', fourcc, 30, (640,480))

    def get_embedding(model, face_pixels):
        # scale pixel values
        face_pixels = face_pixels.astype('float32')
        # standardize pixel values across channels (global)
        mean, std = face_pixels.mean(), face_pixels.std()
        face_pixels = (face_pixels - mean) / std
        # transform face into one sample
        samples = expand_dims(face_pixels, axis=0)
        # make prediction to get embedding
        yhat = model.predict(samples)
        return yhat[0]

    def prediction(model, facenet_model, face):
        face_embed = get_embedding(facenet_model, face)
        in_encoder = Normalizer(norm='l2')
        encoded_face = in_encoder.transform([face_embed])
        id = model.predict(encoded_face)
        prob = model.predict_proba(encoded_face)
        print(id, prob)
        return id[0], prob[0, id - 1] * 100

    model = joblib.load(g_path + '/Recognizer.sav')
    facenet_model = load_model(g_path + '/FaceDetect/facenet_keras.h5')
    cascadePath = g_path + '/FaceDetect/Cascades/haarcascade_frontalface_default.xml'
    faceCascade = cv2.CascadeClassifier(cascadePath)
    font = cv2.FONT_HERSHEY_SIMPLEX

    names = {0: 'Unknown'}
    sus_ids = []
    for i in Susinfo.objects.all():
        names[int(i.susid)] = i.name
        if i.isSuspect == 'yes':
            sus_ids.append(int(i.susid))
    print(names, sus_ids)
    sus_detected = []
    non_sus_detected = []

    cam = cv2.VideoCapture(0)
    cam.set(3, 640)  # set video widht
    cam.set(4, 480)  # set video height
    # Define min window size to be recognized as a face
    minW = 0.1 * cam.get(3)
    minH = 0.1 * cam.get(4)

    while True:
        ret, img = cam.read()
        img = cv2.flip(img, 1)  # Flip vertically
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(int(minW), int(minH)), )
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            id, prob = prediction(model, facenet_model, cv2.resize(img[y:y + h, x:x + w], (160, 160)))

            # If confidence is less them 100 ==> "0" : perfect match
            if (prob < 60):
                id = 0

            if id in sus_ids:
                if id not in sus_detected:
                    acc_sid = "AC361b640ba71f76a616d917cd84e6fbac"
                    auth = ""
                    client = Client(acc_sid, auth)
                    auth_no = '+91' + Susinfo.objects.get(susid=str(id)).phone
                    client.messages.create(from_="+18329063590",
                                           body="!!! The " + names[id] + "-" + str(id) + " is spotted !!!",
                                           to=auth_no)

                    sus_detected.append(id)
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

            else:
                if id not in non_sus_detected:
                    if id != 0:
                        non_sus_detected.append(id)
            if id == 0:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(img, names[id] + ':' + str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
            cv2.putText(img, str(prob), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

        cv2.imshow('camera', img)
        k = cv2.waitKey(10) & 0xff  # Press 'ESC' for exiting video
        if k == 27:
            break
    # Do a bit of cleanup
    print("\n [INFO] Exiting Program and cleanup stuff")
    sus_details = {}
    non_sus_details = {}
    for i in sus_detected:
        sus_details[i] = names[i]
    for i in non_sus_detected:
        non_sus_details[i] = names[i]
    cam.release()
    cv2.destroyAllWindows()

    di = {'sus_detected': sus_details, 'non_sus_detected': non_sus_details,
          'tot': len(sus_detected) + len(non_sus_detected), 'sus': len(sus_detected), 'non_sus': len(non_sus_detected)}
    return render(request, 'FaceDetect/recogniseresult.html', context=di)
