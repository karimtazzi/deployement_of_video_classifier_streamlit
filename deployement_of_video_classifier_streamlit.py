# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 19:15:01 2021

@author: siddhardhan
"""

import numpy as np
import pickle
import streamlit as st
MAX_SEQ_LENGTH = 20
NUM_FEATURES = 2048

# loading the saved model
loaded_model = pickle.load(open('C:\Users\Hinnovis\Downloads\Video_classifier_CNN_RNN-main\Video_classifier_CNN_RNN-main', 'rb'))



def prepare_single_video(frames):
    frames = frames[None, ...]
    frame_mask = np.zeros(shape=(1, MAX_SEQ_LENGTH,), dtype="bool")
    frame_features = np.zeros(shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")

    for i, batch in enumerate(frames):
        video_length = batch.shape[0]
        length = min(MAX_SEQ_LENGTH, video_length)
        for j in range(length):
            frame_features[i, j, :] = feature_extractor.predict(batch[None, j, :])
        frame_mask[i, :length] = 1  # 1 = not masked, 0 = masked

    return frame_features, frame_mask


def sequence_prediction(path):
    class_vocab = label_processor.get_vocabulary()

    frames = load_video(os.path.join("test", path))
    frame_features, frame_mask = prepare_single_video(frames)
    probabilities = sequence_model.predict([frame_features, frame_mask])[0]

    for i in np.argsort(probabilities)[::-1]:
        print(f"  {class_vocab[i]}: {probabilities[i] * 100:5.2f}%")
    return frames
    
def main():
    
    
    # giving a title
    st.title('Video Classifier Web App')
    
    
    # getting the input data from the user
    
    
   video_file = open('C:\Users\Hinnovis\Downloads\Video1', 'rb') #enter the filename with filepath
   video_1 = video_file.read() #reading the file
   st.video(video_1) #displaying the video
    
    
    
    # creating a button for Prediction
    
    if st.button('Diabetes Test Result'):
       test_frames = sequence_prediction(test_video)
        
        
    st.success(test_frames)
    
    
    
    
    
if __name__ == '__main__':
    main()
    