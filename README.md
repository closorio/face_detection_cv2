# **Face_Detection**
Detección de rostros en tiempo real usando una cámara web. 

### Preparación del entorno
#### venv

    $ python3.10 -m venv venv

    Windows	.\venv\Scripts\activate
    Linux/macOS	source venv/bin/activate

    $ pip install -r requirements.txt 

#### Conda


    conda create -n face_emotion_env python=3.10 -y
    conda activate face_emotion_env

    $ pip install -r requirements.txt 

    o

    $ pip install tensorflow==2.16.1 keras==2.16.0 opencv-python==4.8.1.78 imutils==0.5.4 matplotlib==3.8.0 numpy==1.26.0 h5py==3.10.0


    
### Usando WebCam

Ejecutar el archivo FaceEmotionVideo.py

    $ python FaceEmotionVideo.py



### Basado en el proyecto de David Revelo Luna
https://github.com/DavidReveloLuna/Face_Emotion/tree/master 