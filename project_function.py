### Function

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os
import noisereduce as nr
import librosa
import librosa.display
from scipy.io.wavfile import write
import soundfile as sf


def cloudy_effect(img) :

    img_size = img.shape
    img = np.float_(img)
    #print(img_size)
    R_im = img[:, :, 2]
    G_im = img[:, :, 1]
    B_im = img[:, :, 0]
    Y = (0.299 * R_im + 0.578 * G_im + 0.114 * B_im)
    Cb = 0.564 * (B_im - Y) * 0.5
    Cr = 0.713 * (R_im - Y) * 0.5
    result = np.zeros(img_size, np.uint8)
    result[:, :, 2] = (Y + 1.402 * Cr)
    result[:, :, 1] = (Y - 0.344 * Cb - 0.714 * Cr)
    result[:, :, 0] = (Y + 1.772 * Cb)

    return result


def edge_effect(img, img_2, y) :

    # energy map
    img = np.float_(img)
    R_im = img[:, :, 2]
    G_im = img[:, :, 1]
    B_im = img[:, :, 0]
    gray = (0.299 * R_im + 0.578 * G_im + 0.114 * B_im)
    Y = cv.GaussianBlur(gray, (25, 25), 0)
    dx = np.array([(3, 0, -3), (10, 0, -10), (3, 0, -3)])
    dy = np.transpose(dx)
    Ix = cv.filter2D(Y, -1, dx)
    Iy = cv.filter2D(Y, -1, dy)
    res = abs(Ix) + abs(Iy)
    res = np.uint8(res)

    # find block
    img_size_1 = img.shape
    img_size_2 = img_2.shape
    x_r = img_size_1[1] - img_size_2[1]
    y_b = y + img_size_2[0]
    energy_min = 256 * img_size_1[0] * img_size_1[1]
    min_1 = 0
    for j in range(0, x_r) :
        x_2 = j + img_size_2[1]
        energy = np.sum(res[y:y_b, j:x_2])
        if energy_min > energy :
            min_1 = j
            energy_min = energy
    
    return min_1


def muse_image(img_1, img_2, x, y, mode) :
     
    img_1 = np.float_(img_1)
    img_2 = np.float_(img_2)
    img_size_1 = img_1.shape
    img_size_2 = img_2.shape
    result = np.zeros(img_size_1)
    x_r = x + img_size_2[1]
    y_b = y + img_size_2[0]
    print(img_size_1)
    print(img_size_2)
    print(img_size_1[0])
    print(img_size_1[1])

    for i in range(0, img_size_1[1]) :
        for j in range(0, img_size_1[0]) :

            x_2 = i - x
            y_2 = j - y

            # mode 1: for rain
            if mode == 1 :
                if (i >= x) and (i < x_r) and (j >= y) and (j < y_b) :
                    result[j, i, :] = img_1[j, i, :] * 0.6 + img_2[y_2, x_2, :] * 0.4
                else :
                    result[j, i, :] = img_1[j, i, :]

            # mode 2: for object with BLACK background
            elif mode == 2 :
                if (i >= x) and (i < x_r) and (j >= y) and (j < y_b) :
                    pixel = img_2[y_2, x_2, :]
                    energy = np.sum(pixel)
                    if energy <= 10 :
                        result[j, i, :] = img_1[j, i, :]
                    else :
                        result[j, i, :] = img_2[y_2, x_2, :] 
                else :
                    result[j, i, :] = img_1[j, i, :]
            
            # mode 3: for object with WHITE background
            elif mode == 3 :
                if (i >= x) and (i < x_r) and (j >= y) and (j < y_b) :
                    pixel = img_2[y_2, x_2, :]
                    energy = np.sum(pixel)
                    if energy >= 760 :
                        result[j, i, :] = img_1[j, i, :]
                    else :
                        result[j, i, :] = img_2[y_2, x_2, :] 
                else :
                    result[j, i, :] = img_1[j, i, :]
            
            # default: for lightning
            else :
                if (i >= x) and (i < x_r) and (j >= y) and (j < y_b) :
                    result[j, i, :] = img_1[j, i, :] * 0.6 + img_2[y_2, x_2, :] * 0.4
                else :
                    result[j, i, :] = img_1[j, i, :] * 0.6
    
    result = np.uint8(result)
    return result

            

def feat_extraction(path):

    # parameters setting
    signal, sr = librosa.load(path)
    signal = (signal*5)
    reduced_noise = nr.reduce_noise(y=signal, sr=sr)
    frame_length = 32                     # Frame length (samples)
    frame_step = 128                      # Step length (samples)
    num_FFT = frame_length                # FFT freq-quantization
    signal_length = len(signal)           # Signal length
    num_frames = 1 + int(np.ceil((1.0 * signal_length - frame_length) / frame_step))

    # extract feature
    #LPC_array = librosa.lpc(reduced_noise, 5)
    alter_audio = pre_emphasis(signal, coefficient = 0.7)
    alter_STFT = STFT(alter_audio, num_frames, num_FFT, frame_step, frame_length, signal_length, verbose=False)
    pre_audio = pre_emphasis(signal, coefficient = 0.7)
    MFCC_array = librosa.feature.mfcc(pre_audio, sr, S=None, n_mfcc=32, n_fft = 512, n_mels=48)
    RMS_array = librosa.feature.rms(signal, frame_length=frame_length)
    RMS_mean = np.mean(RMS_array, 1)
    RMS_std = np.std(RMS_array, 1)
    freq_mean = np.mean(alter_STFT)
    freq_std  = np.std(alter_STFT)
    result_mean = np.mean(MFCC_array, 1)
    result_std  = np.std(MFCC_array, 1)  
    S_2 = np.abs(librosa.fft_frequencies(sr, n_fft=512))
    S_db = librosa.amplitude_to_db(S_2, ref=np.max)  
    result = np.hstack([result_mean, result_std, freq_mean, freq_std, RMS_mean, RMS_std, S_db])
    #write('./alter.wav', sr, reduced_noise)

    return result.T




def pre_emphasis(signal, coefficient = 0.95):

    return np.append(signal[0], signal[1:] - coefficient*signal[:-1])


# Short Time Fourier Transform
def STFT(time_signal, num_frames, num_FFT, frame_step, frame_length, signal_length, verbose=False):
    padding_length = int((num_frames - 1) * frame_step + frame_length)
    padding_zeros = np.zeros((padding_length - signal_length,))
    padded_signal = np.concatenate((time_signal, padding_zeros))

    # split into frames
    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames*frame_step, frame_step), (frame_length, 1)).T
    indices = np.array(indices,dtype=np.int32)

    # slice signal into frames
    frames = padded_signal[indices]
    # apply window to the signal
    frames *= np.hamming(frame_length)

    # FFT
    complex_spectrum = np.fft.rfft(frames, num_FFT).T
    #print(complex_spectrum.shape)
    absolute_spectrum = np.abs(complex_spectrum)
    
    if verbose:
        print('Signal length :{} samples.'.format(signal_length))
        print('Frame length: {} samples.'.format(frame_length))
        print('Frame step  : {} samples.'.format(frame_step))
        print('Number of frames: {}.'.format(len(frames)))
        print('Shape after FFT: {}.'.format(absolute_spectrum.shape))

    return absolute_spectrum
