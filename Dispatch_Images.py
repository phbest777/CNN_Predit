import os
import random
import shutil


def moveFile(fileDir, trainDir,rate1):
    pathDir = os.listdir(fileDir)  # 取图片的原始路径
    filenumber = len(pathDir)
    rate1 = rate1  # 自定义抽取csv文件的比例，比方说100张抽80张，那就是0.8
    picknumber1 = int(filenumber * rate1)  # 按照rate比例从文件夹中取一定数量的文件
    sample1 = random.sample(pathDir, picknumber1)  # 随机选取picknumber数量的样本
    print(sample1)
    for name in sample1:
        shutil.move(fileDir + name, trainDir + "\\" + name)
    # for i in pathDir:
    #     for j in sample1:
    #         if(i!=j):
    #             shutil.move(fileDir + name, testDir + "\\" + name)

    # shutil.copyfile(fileDir + name, tarDir + name)
    return


if __name__ == '__main__':
    fileDir = "D:\PythonProject\AI\PyTorch\CNN_Predit\DATA\IMG\ResNet\\SMALL\\train\T4\\"
    testDir = 'D:\PythonProject\AI\PyTorch\CNN_Predit\DATA\IMG\ResNet\\SMALL\\test\T4\\'
    # testDir ='D/sy/BSCtest'
    moveFile(fileDir, testDir,0.2)
