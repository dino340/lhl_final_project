import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, load_model
import streamlit as st
<<<<<<< HEAD
from streamlit_webrtc import (
    AudioProcessorBase,
    RTCConfiguration,
    VideoProcessorBase,
    WebRtcMode,
    webrtc_streamer,
)

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

class VideoProcessor(VideoProcessorBase):
    delay = DEFAULT_DELAY
=======
>>>>>>> parent of 60749ca (update to streamlit-webrtc)

st.write("""
# Facemask detector
This application was created by Garrett Brezsnyak for my capstone project for Lighthouse Labs Data Science Program.

It utilizes a CNN model based on VGG16 trained using images from the FFHQ dataset and from MaskedFace-Net.

FFHQ dataset: https://www.kaggle.com/arnaud58/flickrfaceshq-dataset-ffhq

MaskedFace-Net: https://github.com/cabani/MaskedFace-Net""")

# Load most successful model from disk
model = load_model('model-004.model')

# label dictionary for each image class and color dictionary for bounding boxes
labels_dict = {0:'correctly_worn', 1:'incorrectly_worn', 2:'no_mask'}
color_dict = {0:(0,255,0), 1:(255,255,0), 2:(255,0,0)}

# import face detection program
classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# streamlit requrements to display camera feed within streamlit app
st.title("Facemask Feed")
run = st.checkbox('Run, may take a minute to initialize camera')
FRAME_WINDOW = st.image([])

# start webcam capture and set size for downscaling webcam image
size = 1
webcam = cv2.VideoCapture(0)

# begin a loop for webcam capture and face detection
while run:
<<<<<<< HEAD
    webcam = webrtc_streamer(
        key='mask-detection',
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=VideoProcessor,
        async_processing=True
    )
    im = webcam
    
#     im=cv2.flip(im,1)  # mirror image horizontally
    im_color= im      # cv2.cvtColor(im, cv2.COLOR_BGR2RGB)  # color correct image for predictions to RGB
    im_color2= im    # cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
=======
    (rval, im) = webcam.read()  # read images from camera
    im=cv2.flip(im,1)  # mirror image horizontally
    im_color=cv2.cvtColor(im, cv2.COLOR_BGR2RGB)  # color correct image for predictions to RGB
    im_color2=cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
>>>>>>> parent of 60749ca (update to streamlit-webrtc)
    
    mini = cv2.resize(im_color, (im_color.shape[1] // size, im_color.shape[0] // size))  # resize image 
    
    faces = classifier.detectMultiScale(mini)  # run classifier on smaller image to detect faces within feed
    
    for f in faces:   # go through faces and run each of them through model
        (x, y, w, h) = [v * size for v in f]  # define position of faces
        face_img = im_color[y-65:y+h+35, x-45:x+w+45]  # save face images for model preds
        resized = cv2.resize(face_img, (224,224))  # resize for model input
        normalized = resized/255.0  # normalize pixel values as 0-1 floats
        reshaped=np.reshape(normalized,(1,224,224,3))  # reshape array for model
        reshaped= np.vstack([reshaped])
        result = model.predict(reshaped)   # run results through model and generate predictions
        
        label = np.argmax(result,axis=1)[0]  # analyze prediction and save most likely class as label
        
        cv2.rectangle(im_color2,(x-40,y-40),(x+40+w,y+h+40),color_dict[label],2)   # draw a rectangle around detected face
        cv2.rectangle(im_color2,(x,y),(x+w,y),color_dict[label],-1)  # draw rectangle for text
        cv2.putText(im_color2,labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)  #write text on top of image pertaining to class
        
    FRAME_WINDOW.image(im_color2)
else:
    st.write('Stopped')
