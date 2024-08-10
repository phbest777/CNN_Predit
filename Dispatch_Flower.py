import scipy.io
import numpy as np
import os
from PIL import Image

labels = scipy.io.loadmat('./imagelabels.mat')
labels = np.array(labels['labels'][0]) - 1
print("labels:", labels)

setid = scipy.io.loadmat('./setid.mat')

validation = np.array(setid['valid'][0]) - 1
np.random.shuffle(validation)

train = np.array(setid['trnid'][0]) - 1
np.random.shuffle(train)

test = np.array(setid['tstid'][0]) - 1
np.random.shuffle(test)

flower_dir = list()
for img in os.listdir("./DATA/IMG/Flower/jpg"):
    flower_dir.append(os.path.join("./DATA/IMG/Flower/jpg", img))
flower_dir.sort()
'''
划分train集
'''
des_folder_train = "./DATA/IMG/Flower/train"  # 该地址可为新建的训练数据集文件夹的相对地址
for tid in train:
    # 打开图片并获取标签
    img = Image.open(flower_dir[tid])
    print(img)
    # print(flower_dir[tid])
    img = img.resize((256, 256), Image.ANTIALIAS)
    lable = labels[tid]
    # print(lable)
    path = flower_dir[tid]
    print("path:", path)
    base_path = os.path.basename(path)
    print("base_path:", base_path)
    classes = "c" + str(lable)
    class_path = os.path.join(des_folder_train, classes)
    # 判断结果
    if not os.path.exists(class_path):
        os.makedirs(class_path)
    print("class_path:", class_path)
    despath = os.path.join(class_path, base_path)
    print("despath:", despath)
    img.save(despath)

'''
划分valid集
'''
des_folder_validation = "./DATA/IMG/Flower/valid"  # 该地址为新建的验证数据集文件夹的相对地址

for tid in validation:
    img = Image.open(flower_dir[tid])
    # print(flower_dir[tid])
    img = img.resize((256, 256), Image.ANTIALIAS)
    lable = labels[tid]
    # print(lable)
    path = flower_dir[tid]
    print("path:", path)
    base_path = os.path.basename(path)
    print("base_path:", base_path)
    classes = "c" + str(lable)
    class_path = os.path.join(des_folder_validation, classes)
    # 判断结果
    if not os.path.exists(class_path):
        os.makedirs(class_path)
    print("class_path:", class_path)
    despath = os.path.join(class_path, base_path)
    print("despath:", despath)
    img.save(despath)

'''
划分test集
'''
des_folder_test = "./DATA/IMG/Flower/test"  # 该地址为新建的测试数据集文件夹的绝对地址
for tid in test:
    img = Image.open(flower_dir[tid])
    # print(flower_dir[tid])
    img = img.resize((256, 256), Image.ANTIALIAS)
    lable = labels[tid]
    # print(lable)
    path = flower_dir[tid]
    print("path:", path)
    base_path = os.path.basename(path)
    print("base_path:", base_path)
    classes = "c" + str(lable)
    class_path = os.path.join(des_folder_test, classes)
    # 判断结果
    if not os.path.exists(class_path):
        os.makedirs(class_path)
    print("class_path:", class_path)
    despath = os.path.join(class_path, base_path)
    print("despath:", despath)
    img.save(despath)


