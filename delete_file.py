import os  # 引入操作系统模块
import sys  # 用于标准输入输出


def search(path, name):
    for root, dirs, files in os.walk(path):  # path 为根目录
        if name in dirs or name in files:
            flag = 1  # 判断是否找到文件
            root = str(root)
            dirs = str(dirs)
            return os.path.join(root, dirs)
    return -1


path = "./"
name = ""
answer = search(path, name)
if answer == -1:
    print("查无此文件")
else:
    print(answer)
