
import matplotlib.pyplot as plt
import cv2
import imutils
import dlib 
foto = cv2.imread("289070372-358234646380140-3110351821553297969-n.webp")
fotorgb = cv2.cvtColor(foto, cv2.COLOR_BGR2RGB)
plt.imshow(fotorgb)
plt.show()

print(foto.shape)

fotocinza = cv2.cvtColor(foto, cv2.COLOR_BGR2GRAY)
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

print(fotocinza.shape)

facesdetectadas = detector.detectMultiScale(
    fotocinza, scaleFactor = 1.15, minNeighbors = 5,
    minSize = (20, 20), flags = cv2.CASCADE_SCALE_IMAGE)
print("Eu encontrei {} faces nesta foto".format(len(facesdetectadas)))

for(x1, y1, x2, y2) in facesdetectadas:
  cv2.rectangle(fotorgb, (x1, y1), (x1+x2, y1+y2), (0,255, 255), 5)
plt.imshow(fotorgb)
plt.show()


detectorcnn = dlib.cnn_face_detection_model_v1("./mmod_human_face_detector.dat")
foto = cv2.imread('fotoFamilia.jpg')
fotorgb = cv2.cvtColor(foto, cv2.COLOR_BGR2RGB)
facesdetectadascnn = detectorcnn(fotorgb, 1)
print("Eu encontrei {} faces nesta foto".format(len(facesdetectadascnn)))


