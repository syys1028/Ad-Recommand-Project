import os, glob
import numpy as np
import cv2
import random
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split


class ClassificationAndLabeling:
    def __init__(self):
        self.name = 'Image Classification And Labeling'
        self.obs_Xdata = []
        self.obs_Ylabel = []
        self.image_size = (128, 128)
        self.image_folder = 6
        self.path = os.getcwd()  # Updated to work in notebooks or any environment

    def basicCL(self):
        for i in range(0, self.image_folder):
            path_obs = self.path + '/obs' + str(i)
            obs_img_files = glob.glob(path_obs + "/*.jpg")
            for j in obs_img_files:
                obs_img = cv2.imread(j)
                obs_img = cv2.resize(obs_img, self.image_size)
                self.obs_Xdata.append(obs_img)
                self.obs_Ylabel.append(i)
        return self.obs_Xdata, self.obs_Ylabel

    def idgCL(self):
        train_datagen = ImageDataGenerator(rescale=1./255,
                                           rotation_range=30,
                                           width_shift_range=0.1,
                                           height_shift_range=0.1,
                                           shear_range=0.2,
                                           zoom_range=0.2,
                                           horizontal_flip=False,
                                           vertical_flip=False,
                                           fill_mode='nearest')

        train_generator = train_datagen.flow_from_directory(self.path + '/train',
                                                            target_size=(128, 128),
                                                            batch_size=10,
                                                            class_mode='categorical')

        test_datagen = ImageDataGenerator(rescale=1./255)
        test_generator = test_datagen.flow_from_directory(self.path + '/test',
                                                          target_size=(128, 128),
                                                          batch_size=10,
                                                          class_mode='categorical')
        return train_generator, test_generator


class myCallback(Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('loss') < 0.05:
            print('\n Stop training.')
            self.model.stop_training = True


class ODmodel:
    def __init__(self):
        self.name = 'Obstacle detection model'
        self.numofClasses = 6
        self.ODnet = self.ODnetwork()

    def DataPreprocessing(self, obs_Xdata, obs_Ylabel):
        imagedata = np.array(obs_Xdata)
        imagelabel = np.array(obs_Ylabel)
        train_images, test_images, train_labels, test_labels = train_test_split(imagedata, imagelabel, test_size=0.2)
        self.train_images = train_images.astype(float) / 255.0
        self.test_images = test_images.astype(float) / 255.0
        self.train_labels = to_categorical(train_labels)
        self.test_labels = to_categorical(test_labels)

    def ODnetwork(self):
        model = Sequential()
        model.add(Conv2D(32, (5, 5), activation='relu', input_shape=(128, 128, 3)))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.numofClasses, activation='softmax'))
        model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

        return model

    def learn_OD_idgCL(self, trainGenSet, testGenSet, callbacks):
        self.ODnet.fit(trainGenSet, steps_per_epoch=3, epochs=15, validation_data=testGenSet,
                       validation_steps=3, callbacks=[callbacks])
        scores = self.ODnet.evaluate(testGenSet)
        print('accuracy: ' + str(scores[1]))
        print('loss: ' + str(scores[0]))
        self.ODnet.save('obs_model.keras') 


class recommendProduct:
    def __init__(self):
        self.obs_Xdata = []
        self.obs_Ylabel = []
        self.path = os.getcwd()  
        self.image_size = (128, 128)

    def load_images(self, folder_name):
        self.obs_Xdata = []
        obs_img_files = glob.glob(self.path + f"/{folder_name}/*.jpg")
        for img_file in obs_img_files:
            img = cv2.imread(img_file)
            img = cv2.resize(img, self.image_size)
            self.obs_Xdata.append(img)

    def man(self):
        self.load_images('obs0')  
        self.display_images(3)    

    def women(self):
        self.load_images('obs1')  
        self.display_images(3)    

    def man_clothes(self):
        self.load_images('obs0')  
        self.display_images(1)    

    def women_clothes(self):
        self.load_images('obs1')  
        self.display_images(1)    

    def food(self):
        self.load_images('obs2')  
        self.display_images(1)    

    def beauty(self):
        self.load_images('obs3')  
        self.display_images(1)    

    def game(self):
        self.load_images('obs4') 
        self.display_images(1)   

    def hobby(self):
        self.load_images('obs5')  
        self.display_images(1)   

    def display_images(self, count):
        alist = []
        for j in range(0, count):  
            a = random.randrange(0, len(self.obs_Xdata))
            while a in alist:  
                a = random.randrange(0, len(self.obs_Xdata))
            plt.figure()
            plt.imshow(cv2.cvtColor(self.obs_Xdata[a], cv2.COLOR_BGR2RGB))
            plt.title(f"Recommended Product {j+1}")
            plt.show()
            alist.append(a)


obs_labels = ["manclothes", "womenclothes", "food", "beauty", "game", "hobby"]

# 정보 입력받기
a = [10, 20, 30, 40]
r = ["옷", "음식", "뷰티", "게임", "취미"]
while True:
    age = int(input('현재 연령대를 입력하세요(10,20,30,40)\n'))
    gender = input('현재 성별을 입력하세요(남성, 여성)\n')
    recommend = input('추천받고 싶은 항목을 입력하세요(옷,음식,뷰티,게임,취미)\n')
    if age in a:
        if gender == "남성":
            if recommend in r:
                break
        elif gender == "여성":
            if recommend in r:
                break
    print('입력 오류입니다. 다시 입력하세요')
    continue

rec = recommendProduct()
CL = ClassificationAndLabeling()
ODmod = ODmodel()
callbacks = myCallback()

traingenerator, testgenerator = CL.idgCL()
ODmod.learn_OD_idgCL(traingenerator, testgenerator, callbacks)

# 구매 상품 구분하기
path = os.getcwd()
new_obs_model = load_model('obs_model.keras')
img_path = path + '/Capture'  # 구매물품 폴더
img_files = glob.glob(img_path + "/*.jpg")
for img in img_files:
    frame = cv2.imread(img)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (128, 128))
    predicted_result = new_obs_model.predict(np.array([frame]))
    obs_predicted = predicted_result[0]
    for i, obsClass in enumerate(obs_predicted):
        print(obs_labels[i], '=', int(obsClass * 100))
    print('Predicted Result =', obs_labels[obs_predicted.argmax()])
    plt.imshow(frame)
    tmp = "Prediction:" + obs_labels[obs_predicted.argmax()]
    plt.title(tmp)
    plt.show()
buy = obs_labels[obs_predicted.argmax()]
print('구매하신 상품은 ' + buy + '입니다.')

print()
# 성별 추천 상품
print('입력하신 ' + gender + ' 추천 상품 3개입니다.')
if gender == "남성":
    rec.man()

elif gender == "여성":
    rec.women()

print()
# 입력한 항목 추천 상품
print('입력하신 ' + recommend + '의 추천 상품 1개입니다.')
if recommend == "옷":
    if gender == "남성":
        rec.man_clothes()
        print('추천 상품의 구매 링크는 https://www.musinsa.com/app/ 입니다.')
    elif gender == "여성":
        rec.women_clothes()
        print('추천 상품의 구매 링크는 https://ririnco.com/ 입니다.')
elif recommend == "음식":
    rec.food()
    print('추천 상품의 구매 링크는 https://www.kurly.com 입니다.')
elif recommend == "뷰티":
    rec.beauty()
    print('추천 상품의 구매 링크는 https://stylenanda.com/category/3ce/1784/ 입니다.')
elif recommend == "게임":
    rec.game()
    print('추천 상품의 구매 링크는 https://support.nintendo.co.kr/onlineStore 입니다.')
elif recommend == "취미":
    rec.hobby()
    print('추천 상품의 구매 링크는 https://www.coupang.com/ 입니다.')

print()
# 같은 종류의 상품 추천 및 링크 출력
print('구매하신 ' + buy + '의 추천 상품 1개입니다.')
if buy == "manclothes":
    rec.man_clothes()
    print('추천 상품의 구매 링크는 https://www.musinsa.com/app/ 입니다.')
elif buy == "womenclothes":
    rec.women_clothes()
    print('추천 상품의 구매 링크는 https://ririnco.com/ 입니다.')
elif buy == "food":
    rec.food()
    print('추천 상품의 구매 링크는 https://www.kurly.com 입니다.')
elif buy == "beauty":
    rec.beauty()
    print('추천 상품의 구매 링크는 https://stylenanda.com/category/3ce/1784/ 입니다.')
elif buy == "game":
    rec.game()
    print('추천 상품의 구매 링크는 https://support.nintendo.co.kr/onlineStore 입니다.')
elif buy == "hobby":
    rec.hobby()
    print('추천 상품의 구매 링크는 https://www.coupang.com/ 입니다.')
