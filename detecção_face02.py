import cv2

# carregar o arquivo com as características da face
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# carregar a imagem a ser analisada
img = cv2.imread('289070372-358234646380140-3110351821553297969-n.webp')

# converter a imagem para escala de cinza
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# detectar as faces na imagem
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

# desenhar um retângulo em volta de cada face encontrada
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

# mostrar a imagem com as faces detectadas
cv2.imshow('img', img)
cv2.waitKey()
