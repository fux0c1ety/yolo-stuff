import imgaug.augmenters as iaa
import cv2
import numpy as np
import os
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage  # Исправленный импорт
from imgaug import parameters as iap

# Чтение меток в формате YOLOv8
def read_labels(label_file):
    with open(label_file, 'r') as f:
        labels = [line.strip().split() for line in f.readlines()]
    return np.array(labels, dtype=np.float32)

# Запись меток в формате YOLOv8 с целым числом для класса
def write_labels(labels, label_file):
    with open(label_file, 'w') as f:
        for label in labels:
            # Первое значение (класс) записывается как целое число, остальные остаются как есть
            f.write(f"{int(label[0])} " + " ".join(map(str, label[1:])) + '\n')

# Преобразование YOLO меток в угловые координаты (x1, y1, x2, y2)
def yolo_to_corners(labels, img_w, img_h):
    corners = []
    for label in labels:
        x_center, y_center, w, h = label[1] * img_w, label[2] * img_h, label[3] * img_w, label[4] * img_h
        x1, y1 = x_center - w / 2, y_center - h / 2
        x2, y2 = x_center + w / 2, y_center + h / 2
        corners.append([x1, y1, x2, y2, label[0]])  # добавляем класс как пятый элемент
    return np.array(corners)

# Преобразование угловых координат (x1, y1, x2, y2) обратно в YOLO (x_center, y_center, w, h)
def corners_to_yolo(corners, img_w, img_h):
    yolo_labels = []
    for corner in corners:
        x1, y1, x2, y2, cls = corner
        x_center, y_center = (x1 + x2) / 2, (y1 + y2) / 2
        w, h = x2 - x1, y2 - y1
        yolo_labels.append([int(cls), x_center / img_w, y_center / img_h, w / img_w, h / img_h])
    return np.array(yolo_labels)

# Функция для выполнения аугментации и сохранения изображений и меток
def augment_and_save(image, labels, augmenter, image_save_path, label_save_path):
    h, w = image.shape[:2]

    # Преобразуем метки YOLO в координаты углов
    corners = yolo_to_corners(labels, w, h)

    # Преобразуем углы в объекты BoundingBoxes для imgaug
    bbs = BoundingBoxesOnImage([
        BoundingBox(x1=corner[0], y1=corner[1], x2=corner[2], y2=corner[3])
        for corner in corners
    ], shape=image.shape)

    # Аугментируем изображение и box
    image_aug, bbs_aug = augmenter(image=image, bounding_boxes=bbs)

    # Извлекаем аугментированные boxes
    bbs_aug = bbs_aug.remove_out_of_image().clip_out_of_image()  # Удаляем и обрезаем выходящие за пределы изображения боксы
    corners_aug = np.array([[bb.x1, bb.y1, bb.x2, bb.y2, corners[i][4]] for i, bb in enumerate(bbs_aug.bounding_boxes)])

    # Преобразуем аугментированные углы обратно в формат YOLO
    labels_aug = corners_to_yolo(corners_aug, w, h)

    # Сохраняем
    cv2.imwrite(image_save_path, image_aug)
    write_labels(labels_aug, label_save_path)



# Примеры аугментеров
augmenters = {
    'rotate': iaa.Affine(rotate=(-45, 45)),
    'crop': iaa.Crop(percent=(0.1, 0.3)),
    'contrast': iaa.LinearContrast((0.5, 2.0)),
    'gamma': iaa.GammaContrast(1.5),
    'afine': iaa.Affine(translate_percent={"x": 0.1}, scale=0.8),
    'multiply': iaa.Multiply((1.2, 1.5)),
    'flip': iaa.Fliplr(0.5),
    'gaus': iaa.GaussianBlur(sigma=(0, 0.5)),
    'noize': iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
    'gray': iaa.Grayscale(alpha=(0.0, 1.0)),
    'add': iaa.Add((-10, 10), per_channel=0.5),
    'dropout': iaa.Dropout((0.01, 0.1), per_channel=0.5),
    'elastic': iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25),
    'average': iaa.AverageBlur(k=(2, 7)),
    'median': iaa.MedianBlur(k=(3, 11)),
    'sharpen': iaa.Sharpen(alpha=(0, 0.7), lightness=(0.75, 1.5)),
    'super': iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200)),
    'emboss': iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),
    'invert': iaa.Invert(0.05, per_channel=True),
    'coarse': iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
    'wice': iaa.AddElementwise(iap.Discretize((iap.Beta(0.5, 0.5) * 2 - 1.0) * 32)),
}

for file in os.listdir('anns'):
    image_path = f'imgs/{file[:-4]}.jpg'
    label_path = f'anns/{file}'
    image = cv2.imread(image_path)
    labels = read_labels(label_path)
    # Применение аугментаций и сохранение результатов
    for aug_name, augmenter in augmenters.items():
        aug_image_path = f'imgs2/{file[:-4]}_{aug_name}.png'
        aug_label_path = f'anns2/{file[:-4]}_{aug_name}.txt'

        augment_and_save(image, labels, augmenter, aug_image_path, aug_label_path)
