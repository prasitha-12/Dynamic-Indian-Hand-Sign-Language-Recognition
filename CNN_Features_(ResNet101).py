import cv2
import numpy as np
import matplotlib.pyplot as plt

from keras.applications import ResNet101
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input

import time
start_time = time.time()

csvfile = open("ResNet101_features_LSTM.csv", "w")
model = ResNet101(weights='imagenet')

Final_ResNetFeature = np.zeros((1,1000), np.uint16)

for video_id in range(1,959):
    for frame_number in range(0,45):

        file = "final_extracted_frames/"+str(video_id)+"_"+str(frame_number)+".jpg" # Need to give the path to the file
        print(f"Processing video ID: {video_id}, Frame number: {frame_number}")
        frame = cv2.imread(file)   
        newsize = (224, 224)
        img = cv2.resize(frame, newsize, interpolation = cv2.INTER_AREA)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        features = model.predict(x)
        
        #Final_ResNetFeature = np.concatenate(Final_ResNetFeature, features)
        temp = 0;
        for featuredata in features[0, :]:
            if(temp<1000):
                csvfile.write(str(featuredata)+ ",")
                temp = temp + 1
        
        csvfile.write("\n")
        
    print("--- %s seconds ---" % (time.time() - start_time))
    #print(frame_number)

csvfile.flush()
csvfile.close()