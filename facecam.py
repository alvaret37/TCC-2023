import cv2

# Inicializa o detector de rostos
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Inicializa a webcam
video_capture = cv2.VideoCapture(0)

while True:
    # Captura um quadro do vídeo
    ret, frame = video_capture.read()

    # Converte o quadro para escala de cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detecta os rostos no quadro atual
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Desenha um retângulo em volta dos rostos detectados
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Mostra o quadro atual com os rostos detectados
    cv2.imshow('Video', frame)

    # Pára o loop se o usuário pressionar 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera a webcam e fecha as janelas
video_capture.release()
cv2.destroyAllWindows()
