import pandas as pd
import os
import cv2
from PIL import Image
from torchvision import transforms as T
train_img_path = "/content/database/data/ears/train"
test_img_path = "/content/database/data/ears/test"
labels = {}
randomRotation = T.Compose([
    #T.Grayscale(num_output_channels=3),
    T.RandomRotation(degrees=(0, 60))
])
def randomCrop(w, h):
    randomCrop = T.Compose([
        #T.Grayscale(num_output_channels=3),
        T.RandomCrop(size = (h-20, w-20)),
    ])
    return randomCrop

transfoms = T.Compose([
    #T.Grayscale(num_output_channels=3),
    T.ColorJitter(brightness=0.5),
    T.RandomRotation(degrees=(0, 30)),
    T.RandomPerspective(p=1.0, distortion_scale=0.6),
    T.RandomHorizontalFlip(p=1.0),
    T.RandomVerticalFlip()
])

randomPerspective = T.Compose([
    #T.Grayscale(num_output_channels=3),
    T.RandomPerspective(p=0.8)
])

randomBlur = T.Compose([
    #T.Grayscale(num_output_channels=3),
    T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))
])


def transform_images(path, test, labels):
    type_path = "train/"
    if test:
        type_path = "test/"
    save_path = "D:/Faks/Magisterij/Prvi-letnik/Slikovna biometrija/Naloga2/dataset/train2/" 
    load_path = path + type_path
    for file in os.listdir(load_path):
        image = Image.open(os.path.join(load_path, file)).convert("RGB")
        idx = labels[type_path + file]
        print(file)
        image.save(save_path + str(idx) + "/" + file)
        if not test:
            '''w, h = image.size
            img_rand_rotation1 = randomRotation(image)
            rC = randomCrop(w, h)
            if w > 30 and h > 30:
                img_rand_crop1 = rC(image)
                img_rand_crop1.save(save_path + str(idx) + "/" + file.replace(".", "_rand_crop."))
            img_rand_jitter1 = colorJitter(image)
            img_rand_perspective1 = randomPerspective(image)
            img_rand_blur1 = randomBlur(image)


            img_rand_rotation1.save(save_path + str(idx) + "/" + file.replace(".", "_rand_rotation."))
            img_rand_jitter1.save(save_path + str(idx) + "/" + file.replace(".", "_rand_jitter."))
            img_rand_perspective1.save(save_path + str(idx) + "/" + file.replace(".", "_rand_perspective."))
            img_rand_blur1.save(save_path + str(idx) + "/" + file.replace(".", "_rand_blur."))'''

            img1 = transfoms(image)
            img2 = transfoms(image)

            img1.save(save_path + str(idx) + "/" + file.replace(".", "_1."))
            img2.save(save_path + str(idx) + "/" + file.replace(".", "_2."))


if __name__ == "__main__":
    for folder in range(1, 101):
        folder = "D:/Faks/Magisterij/Prvi-letnik/Slikovna biometrija/Naloga2/dataset/detected/train2/" + str(folder)
        os.mkdir(folder)
    label_path = "D:/Faks/Magisterij/Prvi-letnik/Slikovna biometrija/Naloga2/data/perfectly_detected_ears/annotations/recognition/ids.csv"
    df = pd.read_csv(label_path)
    for i in df.values:
        labels[i[0]] = i[1]
    image_path = "D:/Faks/Magisterij/Prvi-letnik/Slikovna biometrija/Naloga2/dataset/detected/"
    transform_images(image_path, False, labels)


