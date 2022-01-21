# for testing 

from hashlib import sha3_384
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf
import pickle
import noisereduce as nr
import librosa
from scipy.io.wavfile import write
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from project_function import cloudy_effect, edge_effect, muse_image, feat_extraction

# parameter setting
#WHITE = (255, 255, 255)
#BLACK = (0, 0, 0)
#GREEN = (0, 255, 0)
#RED = (255, 0, 0)
#BLUE = (0, 0, 255)
#
## object size
#WIDTH = 1280
#HEIGHT = 720
#LIGHTNING_1 = (400, 320)
#LIGHTNING_2 = (760, 250)
#CAT = (200, 250)
#RAIN = (WIDTH, HEIGHT)
#
#
##===== read / write image =====
## load file
#img = cv.imread('./data/origin_2.png')
#img = cv.resize(img, (WIDTH, HEIGHT), interpolation = cv.INTER_CUBIC)
#result = cloudy_effect(img)
##edge_result = edge_effect(img)
#
#### rainy
#rain_1 = cv.imread('./data/rain_1.png')
#rain_2 = cv.imread('./data/rain_2.png')
#rain_1_res = muse_image(result, rain_1, 0, 0, 1)
#rain_2_res = muse_image(result, rain_2, 0, 0, 1)
###
### lightning
#lightning_1 = cv.imread('./data/lightning_1.png')
#lightning_2 = cv.imread('./data/lightning_3.png')
#sparse_x1 = edge_effect(img, lightning_1, 0)
#lightning_1_res = muse_image(result, lightning_1, sparse_x1, 0, 4)
#sparse_x2 = edge_effect(lightning_1_res, lightning_2, 0)
#lightning_2_res = muse_image(result, lightning_2, sparse_x2, 0, 4)
#
#sz = lightning_1.shape
#cv.rectangle(result, (sparse_x1, 0), (sparse_x1+sz[1], sz[0]), BLUE, thickness = 3)
#sz = lightning_2.shape
#cv.rectangle(result, (sparse_x2, 0), (sparse_x2+sz[1], sz[0]), RED, thickness = 3)
#
## cat
#cat_1 = cv.imread('./data/cat_1.png')
#cat_2 = cv.imread('./data/cat_2.png')
#cat_3 = cv.imread('./data/cat_3.png')
#dog_1 = cv.imread('./data/dog_1.png')
#dog_2 = cv.imread('./data/dog_2.png')
##cat_1_res = muse_image(img, cat_1, 100, 450, 3)
##cat_2_res = muse_image(cat_1_res, cat_2, 400, 450, 3)
##cat_3_res = muse_image(cat_2_res, cat_3, 700, 450, 3)
#
#sz = dog_1.shape
#sparse_x1 = edge_effect(img, dog_1, HEIGHT-sz[0])
#dog_res = muse_image(img, dog_1, sparse_x1, HEIGHT-sz[0], 3)
#
#sz = cat_1.shape
#sparse_x2 = edge_effect(dog_res, cat_1, HEIGHT-sz[0])
#cat_res = muse_image(dog_res, cat_1, sparse_x2, HEIGHT-sz[0], 3)
#
#sz = dog_1.shape
#cv.rectangle(img, (sparse_x1, HEIGHT-sz[0]), (sparse_x1+sz[1], HEIGHT), BLUE, thickness = 3)
#sz = cat_1.shape
#cv.rectangle(img, (sparse_x2, HEIGHT-sz[0]), (sparse_x2+sz[1], HEIGHT), RED, thickness = 3)

# find block


## display imaage
#cv.imshow('image', result)
#print("showing...")
#k = cv.waitKey(0)
#print("wait end...")
##
#cv.imshow('image', dog_res)
#print("showing...")
#k = cv.waitKey(0)
#print("wait end...")
#
#cv.imshow('image', cat_res)
#print("showing...")
#k = cv.waitKey(0)
#print("wait end...")
#
#
# write image
#if k == ord("s") :
#    print("writing...")
#    cv.imwrite("./data/testing.png", img)
#
#print("end...")


#===== video =====
#cap = cv.VideoCapture(0)
#if not cap.isOpened():
#    print("Cannot open camera")
#    exit()
#while True:
#    # Capture frame-by-frame
#    ret, frame = cap.read()
#    # if frame is read correctly ret is True
#    if not ret:
#        print("Can't receive frame (stream end?). Exiting ...")
#        break
#    # Our operations on the frame come here
#    #frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
#    # Display the resulting frame
#    frame = edge_effect(frame)
#    cv.imshow('frame', frame)
#    k = cv.waitKey(20)
#    if k == ord('q'):
#        break
## When everything done, release the capture
#cap.release()
#cv.destroyAllWindows()


#===== drawing =====
#img = np.zeros((512, 512, 3), np.uint8)
#cv.line(img, (0, 0), (256, 512), RED, 5)
#cv.rectangle(img, (100, 0),(500, 200), BLUE, 3)
#cv.circle(img, (400, 40), 40, GREEN, -1)
#cv.ellipse(img,(256,256),(100,50),0,0,180,WHITE,-1)
#
#img_2 = np.zeros((512, 512, 3), np.uint8)
##img_2[:,:,0] = img[:,:,0]
##img_2[:,:,1] = img[:,:,0]
#img_2[:,:,2] = img[:,:,0]
#cv.imshow('image', img_2)
#print("showing...")
#k = cv.waitKey(0)


def prediction(sel) :

    #fs = 44100  # Sample rate
    #seconds = 2  # Duration of recording
    #print("record...")
    #myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
    #sd.wait()  # Wait until recording is finished
#
    #print("write...")
    #write('./output.wav', fs, myrecording)  # Save as WAV file 


    print("extraction...")
    #s = feat_extraction("./final/ESC-50-master/audio/1-21189-A-10.wav")
    #s = feat_extraction("./data/clapping_alone1.wav")
    ##ss = feat_extraction("./output.wav")
    #ss = feat_extraction("./data/clapping_alone2.wav")
    #sss = feat_extraction("./output.wav")
    if sel == 0 :
        ss = feat_extraction("./testingaudio/dog_3.wav")
    elif sel == 5 :
        ss = feat_extraction("./testingaudio/cat_3.wav")
    elif sel == 10 :
        ss = feat_extraction("./testingaudio/rain_3.wav")
    elif sel == 19 :
        ss = feat_extraction("./testingaudio/lightning_3.wav")
    elif sel == 22 :
        ss = feat_extraction("./testingaudio/clap_4.wav")
    else :
        ss = feat_extraction("./testingaudio/rain_3.wav")
    #ss = feat_extraction("./output.wav")
    #sss = feat_extraction("./testingaudio/dog_3.wav")
    #s1 = feat_extraction("./testingaudio/cat_1.wav")
    #s2 = feat_extraction("./testingaudio/cat_2.wav")
    #s3 = feat_extraction("./testingaudio/cat_3.wav")
    #s4 = feat_extraction("./testingaudio/cat_4.wav")
    #s5 = feat_extraction("./testingaudio/cat_5.wav")
    #s6 = feat_extraction("./testingaudio/dog_1.wav")
    #s7 = feat_extraction("./testingaudio/dog_2.wav")
    #s8 = feat_extraction("./testingaudio/dog_3.wav")
    #s9 = feat_extraction("./testingaudio/dog_4.wav")
    #s10 = feat_extraction("./testingaudio/dog_5.wav")
    #s11 = feat_extraction("./testingaudio/clap_1.wav")
    #s12 = feat_extraction("./testingaudio/clap_2.wav")
    #s13 = feat_extraction("./testingaudio/clap_3.wav")
    #s14 = feat_extraction("./testingaudio/clap_4.wav")
    #s15 = feat_extraction("./testingaudio/clap_5.wav")
    #s16 = feat_extraction("./testingaudio/lightning_1.wav")
    #s17 = feat_extraction("./testingaudio/lightning_2.wav")
    #s18 = feat_extraction("./testingaudio/lightning_3.wav")
    #s19 = feat_extraction("./testingaudio/lightning_4.wav")
    #s20 = feat_extraction("./testingaudio/lightning_5.wav")
    #s21 = feat_extraction("./testingaudio/rain_1.wav")
    #s22 = feat_extraction("./testingaudio/rain_2.wav")
    #s23 = feat_extraction("./testingaudio/rain_3.wav")
    #s24 = feat_extraction("./testingaudio/rain_4.wav")
    #s25 = feat_extraction("./testingaudio/rain_5.wav")
    #s = np.vstack([s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15, s16, s17, s18, s19, s20, s21, s22, s23, s24, s25])
    s = [ss, ss]
    # load model
    print("loading...")
    filename_1 = './finalized_SVMmodel.sav'
    filename_2 = './finalized_SCmodel.sav'
    SVM_model = pickle.load(open(filename_1, 'rb'))
    SC_model = pickle.load(open(filename_2, 'rb'))

    print("predict...")
    X_test_std_2 = SC_model.transform(s)
    y_predict_cv = SVM_model.predict(X_test_std_2) 
    #print("write...", y_predict_cv)
    #write('output.wav', fs, myrecording)  # Save as WAV file 
    #sound_db = 20 * np.log10(np.max(np.abs(myrecording)))
    #print("sound_db: ", sound_db)
    #if sound_db >= -12 :
    if y_predict_cv[0] == '0' :
        print("predict: dog")
        result = 0
    elif y_predict_cv[0] == '5' :
        print("predict: cat")
        result = 5
    elif y_predict_cv[0] == '10' :
        print("predict: rain")
        result = 10
    elif y_predict_cv[0] == '19' :
        print("predict: thunderstorm")
        result = 19
    elif y_predict_cv[0] == '22' :
        print("predict: clapping")
        result = 22
    else :
        print("predict: ERROR")
        result = 30
    #else :
    #    print("no sound")
    #    result = 36

    return result