import streamlit as st
import pandas as pd
import cv2
from PIL import Image, ImageEnhance
import numpy as np
import os
#import tensorflow as tf
#import tensorflow_hub as hub
import time
import sys
# from streamlit_embedcode import github_gist
import urllib.request
import urllib
# import moviepy.editor as moviepy
from collections import Counter


def object_detection_image():
    st.title('Object Detection for Images')
    st.subheader("""
    This object detection project takes in images and outputs the images with bounding boxes created around the symbols in the image, and more details can be viewed from checkbox provided for each images.
    """)
    files = st.file_uploader('Upload Image', type=['jpg', 'png', 'jpeg'], accept_multiple_files=True)

    if files:
        for i, file in enumerate(files):
            img1 = Image.open(file)
            img2 = np.array(img1)
            st.image(img1, caption="Uploaded Image")
            my_bar = st.progress(0)
            confThreshold = st.slider(f'Confidence_{i}', 0, 100, 50)
            nmsThreshold = st.slider(f'Threshold_{i}', 0, 100, 20)
            classNames = []
            whT = 608
            url = "https://raw.githubusercontent.com/Priyanshu88/newone-streamlit/main/labels/coconames.txt"
            f = urllib.request.urlopen(url)
            classNames = [line.decode('utf-8').strip() for  line in f]
            # f = open(
            #     r'path', 'r')
            # lines = f.readlines()
            # classNames = [line.strip() for line in lines]
            config_path = r'config_n_weights\yolov4-custom.cfg'
            weights_path = r'config_n_weights\yolov4_best.weights'
            net = cv2.dnn.readNet(config_path, weights_path)
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            # net.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)
            # net.setPreferableBackend(cv2.dnn.DNN_TARGET_CUDA)

            def findObjects(outputs, img):
                hT, wT, cT = img2.shape
                bbox = []
                classIds = []
                confs = []
                for output in outputs:
                    for det in output:
                        scores = det[5:]
                        classId = np.argmax(scores)
                        confidence = scores[classId]
                        if confidence > (confThreshold/100):
                            w, h = int(det[2]*wT), int(det[3]*hT)
                            x, y = int((det[0]*wT)-w/2), int((det[1]*hT)-h/2)
                            bbox.append([x, y, w, h])
                            classIds.append(classId)
                            confs.append(float(confidence))
                            label = classNames[classId].upper()
                indices = cv2.dnn.NMSBoxes(
                    bbox, confs, confThreshold/100, nmsThreshold/100)
                obj_list = []
                confi_list = []
                # drawing rectangle around object
                for i in indices:
                    i = i
                    box = bbox[i]
                    x, y, w, h = box[0], box[1], box[2], box[3]
                    # print(x,y,w,h)
                    cv2.rectangle(img2, (x, y), (x+w, y+h), (240, 54, 230), 2)
                    # print(i,confs[i],classIds[i])
                    obj_list.append(classNames[classIds[i]].upper())
                    confi_list.append(int(confs[i]*100))
                    cv2.putText(img2, f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%',
                                (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, .3, (240, 0, 240), 1)
                label_count = Counter(obj_list)
                label_names = label_count.keys()
                label_counts = label_count.values()
                df = pd.DataFrame(list(zip(obj_list, confi_list)), columns=[
                                'Object Name', 'Confidence'])
                df1 = pd.DataFrame({'Object Name': list(
                    label_names), 'Count': list(label_counts)})
                if st.checkbox(f"Show Object's list in {file.name}"):
                    st.write(df)
                if st.checkbox(f"Show label's count in {file.name}"):
                    st.write(df1)
                if st.checkbox(f"Show Confidence bar chart in {file.name}"):
                    st.subheader(
                        f'Bar chart for confidence levels for {file.name}')
                    st.bar_chart(df["Confidence"])
            blob = cv2.dnn.blobFromImage(
                img2, 1 / 255, (whT, whT), [0, 0, 0], swapRB=True, crop=False)
            net.setInput(blob)
            layersNames = net.getLayerNames()
            outputNames = [layersNames[i-1]
                        for i in net.getUnconnectedOutLayers()]
            outputs = net.forward(outputNames)
            findObjects(outputs, img2)
            st.image(img2, caption='Proccesed Image.')
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            my_bar.progress(100)


def main():
    new_title = '<p style="font-size: 42px;">Welcome The Object Detection App!</p>'
    read_me_0 = st.markdown(new_title, unsafe_allow_html=True)
    read_me = st.markdown("""
    The application is built using Streamlit and OpenCV 
    to demonstrate YOLO Object detection in images for identifying the service keys
    in Digital real-world architectural floor plans. It performs detection on multiple images, can count the number of labels and list of labels.
    This YOLO object Detection project can detect 80 objects(i.e classes)
    in image. The full list of the classes can be found 
    [here](https://github.com/).""")
    st.sidebar.title("Select Activity")
    choice = st.sidebar.selectbox(
        "MODE", ("About", "Object Detection(Image)"))
    if choice == "Object Detection(Image)":
        read_me_0.empty()
        read_me.empty()
        object_detection_image()
    elif choice == "About":
        print()


if __name__ == '__main__':
    main()
