# -*- coding:utf-8 -*-
# @Author: Wei Yi

import os
import pickle
import cv2
import numpy as np
import collections


class Information:
    def __init__(self, save_path, load=False, path="Infomation.pkl"):
        if not load:
            self.character_map = {}
            # 字符-序号 映射
            # 结构{'字':[路径,字序号,文件夹内编号]}
            self.processed_img = set()
            self.int_ch_pair = {}
            self.record = collections.defaultdict(int)
            # 已处理过的图片名
        else:
            pkl_file = open(path, 'rb')
            self.character_map = pickle.load(pkl_file)
            self.processed_img = pickle.load(pkl_file)
            self.int_ch_pair = pickle.load(pkl_file)
            self.record = pickle.load(pkl_file)
        if self.character_map != {}:
            self.tot_characters = len(self.character_map.keys())
            # 给新字符确定编号
        else:
            self.tot_characters = 0
        self.save_path = save_path
        if (not (os.path.exists("单字"))):
            os.mkdir("单字")
        self.path = "单字\\" + save_path + "\\"
        if (not (os.path.exists(self.path))):
            os.mkdir(self.path)
        self.openLog()

    def openLog(self):
        self.file = open('LOG.txt', 'a', encoding="utf-8")

    def closeLog(self):
        self.file.close()

    def hasCharater(self, ch):
        has = self.record[ch]
        if has == 0:
            self.record[ch] += 1
            return False
        else:
            return True

    def addCharacter(self, ch):
        path = self.path + ch
        order = self.tot_characters
        self.tot_characters += 1
        if (not (os.path.exists(path))):
            os.mkdir(path)
            # 为新字符创建目录 例：..\\train\\1
        self.character_map[ch] = [path, order, 0]
        self.int_ch_pair[order] = ch

    def writeLog(self, type, message=None, img=None, text=None, name=None):
        if type == "Img":
            self.file.write("WARNING! " + '\t' + name + ' ' + message + '\t' + name + '\n')
        if type == "Col":
            self.file.write("WARNING! This image produces ZERO character!" + '\t' + text + '\n')
            name = self.mistake_folder + text.replace('/', '~') + '.txt'
            with open(name, 'w', encoding="utf-8") as file:
                file.write(text + '\n')
            text = text.replace('/', '~')
            path = self.mistake_folder + text + ".png"
            cv2.imencode('.png', img)[1].tofile(path)
        if type == "Error":
            self.file.write("WARNING! " + '\t' + name + " Unable to process!" + '\t' + name + '\n')

    def save(self):
        output = open(self.path + 'Infomation.pkl', 'wb')
        pickle.dump(self.character_map, output)
        pickle.dump(self.processed_img, output)
        pickle.dump(self.int_ch_pair, output)
        pickle.dump(self.record, output)
        self.closeLog()
