## Detección de emociones en tiempo real ##
# Este script utiliza un modelo de red neuronal convolucional (CNN) para detectar emociones en tiempo real a partir de un video.
# El modelo se basa en la arquitectura VGG16 y ha sido entrenado con un conjunto de datos de imágenes faciales etiquetadas con emociones extraidos de FER2013.
# Import de librerias actualizadas
from keras.applications.imagenet_utils import preprocess_input
from keras.utils import img_to_array
from keras.models import load_model
import numpy as np
import imutils
import cv2
import time


# Variables para calcular FPS
time_actualframe = 0
time_prevframe = 0

# Tipos de emociones del detector
classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Cargamos el modelo de detección de rostros
prototxtPath = "face_detector/deploy.prototxt"
weightsPath = "face_detector/res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# Carga el detector de clasificación de emociones
emotionModel = load_model("./models/VGG16/modelFER.h5")

# Se crea la captura de video
cam = cv2.VideoCapture(0)

def predict_emotion(frame, faceNet, emotionModel):
    # Construye un blob de la imagen con tamaño ajustado para mejor rendimiento
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), 
                               (104.0, 177.0, 123.0))

    # Realiza las detecciones de rostros
    faceNet.setInput(blob)
    detections = faceNet.forward()

    faces = []
    locs = []
    preds = []
    
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        # Filtra detecciones débiles
        if confidence > 0.4:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (Xi, Yi, Xf, Yf) = box.astype("int")

            # Asegura que las coordenadas estén dentro de la imagen
            Xi, Yi = max(0, Xi), max(0, Yi)
            Xf, Yf = min(w - 1, Xf), min(h - 1, Yf)
            
            # Extrae el ROI del rostro, convierte a escala de grises y redimensiona
            face = frame[Yi:Yf, Xi:Xf]
            if face.size == 0:
                continue
                
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            face = cv2.resize(face, (48, 48))
            face = img_to_array(face)
            face = np.expand_dims(face, axis=0)
            face = face / 255.0  # Normalización

            faces.append(face)
            locs.append((Xi, Yi, Xf, Yf))

            # Predicción de emociones
            pred = emotionModel.predict(face, verbose=0)
            preds.append(pred[0])

    return (locs, preds)

while True:
    ret, frame = cam.read()
    if not ret:
        break
        
    frame = imutils.resize(frame, width=640)
    (locs, preds) = predict_emotion(frame, faceNet, emotionModel)
    
    for (box, pred) in zip(locs, preds):
        (Xi, Yi, Xf, Yf) = box
        emotion_idx = np.argmax(pred)
        emotion_prob = pred[emotion_idx]
        
        label = f"{classes[emotion_idx]}: {emotion_prob * 100:.0f}%"
        
        # Dibuja el rectángulo y la etiqueta
        cv2.rectangle(frame, (Xi, Yi-40), (Xf, Yi), (255, 0, 0), -1)
        cv2.putText(frame, label, (Xi+5, Yi-15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.rectangle(frame, (Xi, Yi), (Xf, Yf), (255, 0, 0), 2)

    # Cálculo de FPS
    time_actualframe = time.time()
    if time_actualframe > time_prevframe:
        fps = 1/(time_actualframe - time_prevframe)
    time_prevframe = time_actualframe

    cv2.putText(frame, f"{int(fps)} FPS", (5, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()