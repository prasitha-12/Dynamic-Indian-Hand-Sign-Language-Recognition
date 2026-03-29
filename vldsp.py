import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

import time
start_time = time.time()

#Kirsch Masks
msk = np.zeros((3,3,8))
msk[:,:,0] = np.array([[- 3,- 3,5],[- 3,0,5],[- 3,- 3,5]]) #east

msk[:,:,1] = np.array([[- 3,5,5],[- 3,0,5],[- 3,- 3,- 3]])#northeast

msk[:,:,2] = np.array([[5,5,5],[- 3,0,- 3],[- 3,- 3,- 3]])#north

msk[:,:,3] = np.array([[5,5,- 3],[5,0,- 3],[- 3,- 3,- 3]])#northwest

msk[:,:,4] = np.array([[5,- 3,- 3],[5,0,- 3],[5,- 3,- 3]])#west

msk[:,:,5] = np.array([[- 3,- 3,- 3],[5,0,- 3],[5,5,- 3]])#southwest

msk[:,:,6] = np.array([[- 3,- 3,- 3],[- 3,0,- 3],[5,5,5]])#south

msk[:,:,7] = np.array([[- 3,- 3,- 3],[- 3,0,5],[- 3,5,5]])#southeast

csvfile = open("VLDSPFeatures.csv", "w")
#for each video
for video_id in range(1,959):
    #Final VLDSP Array
    Final_VLDSP = np.zeros((62,1), np.uint8)
    for frame_id in range(0,45):
            # Need to give the path to the file    
            file1 = f"C:/Users/PRASITHA/OneDrive/Desktop/demo_v3/final_extracted_frames/{video_id}_{frame_id}.jpg"
            frame1 = cv2.imread(file1)
            # Need to give the path to the file
            file2 = f"C:/Users/PRASITHA/OneDrive/Desktop/demo_v3/final_extracted_frames/{video_id}_{frame_id+1}.jpg"
            frame2 = cv2.imread(file2)
            # Need to give the path to the file
            file3 = f"C:/Users/PRASITHA/OneDrive/Desktop/demo_v3/final_extracted_frames/{video_id}_{frame_id+2}.jpg"
            frame3 = cv2.imread(file3)
                
            #convert all frame images to grayscale
            frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            frame3_gray = cv2.cvtColor(frame3, cv2.COLOR_BGR2GRAY)

            frame1_gray = cv2.resize(frame1_gray,(224,224), interpolation = cv2.INTER_AREA)
            frame2_gray = cv2.resize(frame2_gray, (224,224), interpolation = cv2.INTER_AREA)
            frame3_gray = cv2.resize(frame3_gray, (224,224), interpolation = cv2.INTER_AREA)
                
            VLDSP_positive = np.zeros((frame1_gray.shape[0], frame1_gray.shape[1]), np.uint16)
            VLDSP_negative = np.zeros((frame1_gray.shape[0], frame1_gray.shape[1]), np.uint16)
                
            #frame mask applied arrays
            indvals = np.zeros((24,1))
                
            r, c = frame1_gray.shape

            for i in range(r-2):
                for j in range(c-2):
                        P1 = frame1_gray[i:i+3, j:j+3]
                        P2 = frame2_gray[i:i+3, j:j+3]
                        P3 = frame3_gray[i:i+3, j:j+3]
                        val1, val2, val3 = np.zeros(8), np.zeros(8), np.zeros(8)
                        for flag in range(8):
                            val1[flag] = np.sum(np.multiply(P1, msk[:, :, flag]))
                            val2[flag] = np.sum(np.multiply(P2, msk[:, :, flag]))
                            val3[flag] = np.sum(np.multiply(P3, msk[:, :, flag]))

                        zzz = np.zeros(24)
                        count = 8
                        for abcd in range(8):
                            zzz[abcd] = val1[abcd]
                        for abcd in range(8):
                            zzz[count] = val2[abcd]
                            count += 1
                        count = 16
                        for abcd in range(8):
                            zzz[count] = val3[abcd]
                            count += 1

                        qqq = np.argsort(zzz)
                        qqq2 = np.zeros(24)
                        for abc in range(24):
                            if (qqq[abc] > 8 and qqq[abc] < 17):
                                qqq2[abc] = qqq[abc] - 8
                            elif (qqq[abc] > 16):
                                qqq2[abc] = qqq[abc] - 16
                            else:
                                qqq2[abc] = qqq[abc]

                        low_index = qqq2[0] 
                        if (qqq2[0] != qqq2[1]):
                            low_index2 = qqq2[1] 
                        elif (qqq2[1] != qqq2[2]):
                            low_index2 = qqq2[2] 
                        else:
                            low_index2 = qqq2[3] 

                        high_index = qqq2[23] 
                        if (qqq2[23] != qqq2[22]):
                            high_index2 = qqq2[22] 
                        elif (qqq2[22] != qqq2[21]):
                            high_index2 = qqq2[21] 
                        else:
                            high_index2 = qqq2[20] 

                        zzz22 = np.zeros(8)
                        if (abs(high_index - high_index2) == 1 and (high_index > high_index2)):
                            S1 = 0
                        elif (abs(high_index - high_index2) == 1 and (high_index < high_index2)):
                            S1 = 1
                        elif(np.abs(high_index - high_index2)==2 and high_index>high_index2 ):
                            S1 = 2;
                        elif(np.abs(high_index - high_index2)==2 and high_index<high_index2 ):
                            S1 = 3;
                        else:
                            S1 = 4
                            
                        if (abs(low_index - low_index2) == 1 and (low_index > low_index2)):
                            S2 = 0
                        elif (abs(low_index - low_index2) == 1 and (low_index < low_index2)):
                            S2 = 1
                        elif(np.abs(low_index - low_index2)==2 and low_index>low_index2 ):
                            S2 = 2;
                        elif(np.abs(low_index - low_index2)==2 and low_index<low_index2 ):
                            S2 = 3;
                        else:
                            S2 = 4
                            
                            
                        if(S1 < 4):
                            LDSP_positive = 6 * high_index + 3 * high_index2 + S1
                        else:
                            LDSP_positive = 62
                        if(S2 < 4):
                            LDSP_negative = 6 * low_index + 3 * low_index2 + S2
                        else:
                            LDSP_negative = 62

                        VLDSP_positive[i,j] =  LDSP_positive

                        VLDSP_negative[i,j] =  LDSP_negative
                        
                VLDSPval = np.concatenate((VLDSP_positive, VLDSP_negative), axis=None)
                        #print("VLDSPval frame number: "+str(frame_number))
                
                #Create histogram for VLDSPval
                VLDSPHist = cv2.calcHist([VLDSPval], [0], None, [62], [0, 62])
                
            print("--- %s seconds ---" % (time.time() - start_time))
            # Final_VLDSP = Final_VLDSP[np.nonzero(Final_VLDSP)]   
            temp = 0;
            for featuredata in VLDSPHist[0:62]:
                if(temp<62):
                    #print(str(featuredata))
                    csvfile.write(str(featuredata)+",")  #62 features
                    temp = temp + 1
            
            csvfile.write("\n")
            csvfile.flush()

csvfile.close()