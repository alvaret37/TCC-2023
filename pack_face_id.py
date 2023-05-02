import cv2
import os

# Caminho para o arquivo XML com o classificador de rostos
face_cascade_path = 'haarcascade_frontalface_default.xml'

# Carrega o classificador de rostos
face_cascade = cv2.CascadeClassifier(face_cascade_path)

# Função para detectar rostos em uma imagem
def detect_faces(image, face_cascade):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    return faces

# Caminho para a pasta com as imagens a serem analisadas
folder_path = 'fotos'

# Percorre todas as imagens na pasta
for filename in os.listdir(folder_path):
    # Caminho completo para o arquivo de imagem
    image_path = os.path.join(folder_path, filename)

    # Carrega a imagem
    image = cv2.imread(image_path)

    # Detecta os rostos na imagem
    faces = detect_faces(image, face_cascade)

    # Desenha um retângulo ao redor de cada rosto detectado
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Salva a imagem com os retângulos dos rostos detectados
    output_path = os.path.join(folder_path, 'processed_' + filename)
    cv2.imwrite(output_path, image)
