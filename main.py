# DSP final project

### Main Function of Final Project
### date: 2022 / 1 / 17


from decimal import Clamped
import pygame
#import random
import os
from statistics import mode
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf
import pickle
from sklearn.svm import SVC, LinearSVC
from scipy.io.wavfile import write
from sklearn.preprocessing import StandardScaler
from testing import prediction
from project_function import cloudy_effect, edge_effect, muse_image, feat_extraction


# parameter setting
FPS = 5
WIDTH = 1280
HEIGHT = 720


# color setting
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)


# object size
LIGHTNING_1 = (400, 320)
LIGHTNING_2 = (760, 250)
CAT = (200, 250)
RAIN = (WIDTH, HEIGHT)


#===== read / write image =====

# load file
print("image loading...")
img = cv.imread('./data/origin_2.png')
rain_1 = cv.imread('./data/rain_1.png')
rain_2 = cv.imread('./data/rain_2.png')
lightning_1 = cv.imread('./data/lightning_1.png')
lightning_2 = cv.imread('./data/lightning_3.png')
cat_1 = cv.imread('./data/cat_1.png')
cat_2 = cv.imread('./data/cat_2.png')
cat_3 = cv.imread('./data/cat_3.png')
dog_1 = cv.imread('./data/dog_1.png')
dog_2 = cv.imread('./data/dog_2.png')


# prepare (rain, lightning)
print("image reconstruct...")
img = cv.resize(img, (WIDTH, HEIGHT), interpolation = cv.INTER_CUBIC)
cloudy_res = cloudy_effect(img)
rain_res1 = muse_image(cloudy_res, rain_1, 0, 0, 1)
rain_res2 = muse_image(cloudy_res, rain_2, 0, 0, 1)
sparse_lightning_x1 = edge_effect(img, lightning_1, 0)
lightning_1_res = muse_image(cloudy_res, lightning_1, sparse_lightning_x1, 0, 4)
sparse_lightning_x2 = edge_effect(lightning_1_res, lightning_2, 0)
lightning_2_res = muse_image(cloudy_res, lightning_2, sparse_lightning_x2, 0, 4)
sz = dog_1.shape
sparse_dog_x = edge_effect(img, dog_1, HEIGHT-sz[0])
sparse_dog_y = HEIGHT-sz[0] + 40
dog_res = muse_image(img, dog_1, sparse_dog_x, HEIGHT-sz[0], 3)
sz = cat_1.shape
sparse_cat_x = edge_effect(dog_res, cat_1, HEIGHT-sz[0])
sparse_cat_y = HEIGHT-sz[0] + 40


# store image
print("writing...")
cv.imwrite("./data/img_res.png", img)
cv.imwrite("./data/rain_res1.png", rain_res1)
cv.imwrite("./data/rain_res2.png", rain_res2)
cv.imwrite("./data/lightning_res1.png", lightning_1_res)
cv.imwrite("./data/lightning_res2.png", lightning_2_res)


# pygame initialize
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT+40))
pygame.display.set_caption("Sound-changing Image")
running = True
clock = pygame.time.Clock()


# image objects
ori_img     = pygame.image.load(os.path.join("data", "img_res.png"))
rain_img_1  = pygame.image.load(os.path.join("data", "rain_res1.png"))
rain_img_2  = pygame.image.load(os.path.join("data", "rain_res2.png"))
light_img_1 = pygame.image.load(os.path.join("data", "lightning_res1.png"))
light_img_2 = pygame.image.load(os.path.join("data", "lightning_res2.png"))
cat_img_1   = pygame.image.load(os.path.join("data", "cat_1.png")).convert()
cat_img_2   = pygame.image.load(os.path.join("data", "cat_2.png")).convert()
cat_img_3   = pygame.image.load(os.path.join("data", "cat_3.png")).convert()
dog_img_1   = pygame.image.load(os.path.join("data", "dog_1.png")).convert()
dog_img_2   = pygame.image.load(os.path.join("data", "dog_2.png")).convert()
dog_img_3   = pygame.image.load(os.path.join("data", "dog_3.png")).convert()
font_name = pygame.font.match_font('arial')


# sound objects
cat_sound = pygame.mixer.Sound(os.path.join("testingaudio", "cat_3.wav"))
dog_sound = pygame.mixer.Sound(os.path.join("testingaudio", "dog_3.wav"))
clap_sound = pygame.mixer.Sound(os.path.join("testingaudio", "clap_4.wav"))
lightning_sound = pygame.mixer.Sound(os.path.join("testingaudio", "lightning_3.wav"))
rain_sound = pygame.mixer.Sound(os.path.join("testingaudio", "rain_3.wav"))

# object setting
all_sprites = pygame.sprite.Group()

def draw_text(surf, text, size, x, y) :
    font = pygame.font.Font(font_name, size)
    text_surface = font.render(text, True, BLACK)
    text_rect = text_surface.get_rect()
    text_rect.x = x
    text_rect.top = y
    surf.blit(text_surface, text_rect)


class Dog(pygame.sprite.Sprite) :
    def __init__(self) :
        pygame.sprite.Sprite.__init__(self)
        self.image = dog_img_1
        self.image.set_colorkey(WHITE)
        self.rect = self.image.get_rect()
        self.rect.x = sparse_dog_x
        self.rect.y = sparse_dog_y

    def update(self, type) :
        if type == 1 :
            self.image = dog_img_1
            self.image.set_colorkey(WHITE)
        elif type == 2 :
            self.image = dog_img_2
            self.image.set_colorkey(WHITE)
        elif type == 3 :
            self.image = dog_img_3
            self.image.set_colorkey(WHITE)

class Cat(pygame.sprite.Sprite) :
    def __init__(self) :
        pygame.sprite.Sprite.__init__(self)
        self.image = cat_img_1
        self.image.set_colorkey(WHITE)
        self.rect = self.image.get_rect()
        self.rect.x = sparse_cat_x
        self.rect.y = sparse_cat_y

    def update(self, type) :
        if type == 1 :
            self.image = cat_img_1
            self.image.set_colorkey(WHITE)
        elif type == 2 :
            self.image = cat_img_2
            self.image.set_colorkey(WHITE)
        elif type == 3 :
            self.image = cat_img_3
            self.image.set_colorkey(WHITE)


# pygame 
# start pygame
dog_mode = 0
dog_cnt = 0
cat_mode = 0
cat_cnt = 0
weather = 0
weather_n = 0
rain_cnt = 0
guess = 1

print("game start...")
while running :

    # sampling rate
    clock.tick(FPS)

    # input
    for event in pygame.event.get() :
        if event.type == pygame.QUIT :
            running = False
        elif event.type == pygame.KEYDOWN :
            #guess = prediction()
            #print(guess) 
            if event.key == pygame.K_d :
            #if guess == 0 :
                print("in")
                dog_sound.play()
                guess = prediction(0)
                dog_cnt = 0
                if dog_mode == 0 :
                    dog = Dog()
                    dog_mode = 1
                    all_sprites.add(dog)
            if event.key == pygame.K_c :
            #if guess == 5 :
                print("in")
                cat_sound.play()
                guess = prediction(5)
                cat_cnt = 0
                if cat_mode == 0 :
                    cat = Cat()
                    cat_mode = 1
                    all_sprites.add(cat)
            if event.key == pygame.K_r :
            #if guess == 10 :
                print("in")
                rain_sound.play()
                guess = prediction(10)
                rain_cnt = 0
                weather = 1
            if event.key == pygame.K_l :
            #if guess == 19 :
                print("in")
                lightning_sound.play()
                guess = prediction(19)
                rain_cnt = 0
                weather = 2
            if event.key == pygame.K_p :
            #if guess == 22 :
                print("in")
                clap_sound.play()
                guess = prediction(22)
                dog_cnt = 0
                cat_cnt = 0
                if dog_mode == 1 :
                    dog.update(2)
                    dog_mode = 2
                elif dog_mode == 2 :
                    dog.update(1)
                    dog_mode = 1
                if cat_mode == 1 :
                    cat.update(2)
                    cat_mode = 2


    # renew
    if dog_mode != 0 :
        dog_cnt += 1
        if dog_cnt > 25 :
            dog.kill()
            dog_mode = 0
            dog_cnt = 0

    if cat_mode != 0 :
        cat_cnt += 1
        if cat_cnt > 25 :
            cat.kill()
            cat_mode = 0
            cat_cnt = 0
        elif (cat_mode == 2) and (cat_cnt >= 2) :
            cat.update(1)
            cat_mode = 1
    

    #show
    screen.fill(WHITE)

    if weather == 1 :
        rain_cnt += 1
        if rain_cnt > 40 :
            weather = 0
            rain_cnt = 0
            screen.blit(ori_img, (0, 40))
        elif (rain_cnt % 2 == 0) :
            screen.blit(rain_img_1, (0, 40))
        else :
            screen.blit(rain_img_2, (0, 40))
    elif weather == 2 :
        rain_cnt += 1
        if (rain_cnt == 1) or (rain_cnt == 2) :
            screen.blit(light_img_2, (0, 40))
        elif rain_cnt == 3 :
            screen.blit(light_img_1, (0, 40))
            if cat_mode != 0 :
                cat.update(3)
            if dog_mode != 0 :
                dog.update(3)
        else :
            weather = 0
            rain_cnt = 0
            if cat_mode != 0 :
                cat.kill()
                cat_mode = 0
            if dog_mode != 0 :
                dog.kill()
                dog_mode = 0
            screen.blit(ori_img, (0, 40))
    else :
        screen.blit(ori_img, (0, 40))
    all_sprites.draw(screen)
    draw_text(screen, 'C: cat    D: dog    L:lightning    R: rain    P: clap', 30, 10, 4)

    if guess == 0 :
        draw_text(screen, "Detect: dog", 36, 700, 2)
    elif guess == 5 :
        draw_text(screen, "Detect: cat", 36, 700, 2)
    elif guess == 10 :
        draw_text(screen, "Detect: rain", 36, 700, 2)
    elif guess == 19 :
        draw_text(screen, "Detect: thunderstorm", 36, 700, 2)
    elif guess == 22 :
        draw_text(screen, "Detect: clapping", 36, 700, 2)
    else :
        draw_text(screen, "Detect: NO SOUND!", 36, 700, 2)

    pygame.display.update()

    # predict
    #guess = prediction()

pygame.quit()


