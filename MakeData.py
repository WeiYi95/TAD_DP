# -*- coding:utf-8 -*-
# @Author: Wei Yi

import cv2
import numpy as np
from Information import Information
from Cutter_Eraser import V_ColumnCutter, V_TextCutter
from TextProcessor import TextProcessor
import random
import time
import re


flag = False


class MakeData:
    def __init__(self, file, path, load=False):
        self.file_list = file
        self.Info = Information(path, load=load)
        self.tot = 0

    def makeData(self):
        for file in self.file_list:
            try:
                try:
                    img_name = file + ".png"
                    text_name = file + ".txt"
                    text_processor = TextProcessor(text_name)
                    text, double_lines, pic_words, ori_t = text_processor.getProcessText()
                    hasKR0008 = False
                    for sent in ori_t:
                        if sent.find("KR0008") != -1 or sent.find("KR0034") != -1 or sent.find(
                                "KR0146") != -1 or sent.find("KR0306") != -1 or sent.find("KR0320") != -1 or sent.find(
                                "KR1320") != -1 or sent.find("KR1350") != -1 or sent.find("KR2853") != -1 or sent.find(
                                "KR3213") != -1 or sent.find("KR3578") != -1 or sent.find("KR3577") != -1 or sent.find(
                                "KR4283") != -1 or sent.find("KR4472") != -1:
                            hasKR0008 = True
                            break
                    if hasKR0008:
                        continue
                    type = text_processor.getType()
                    col_cutter = V_ColumnCutter(img_name, type)
                    cols, img_with_line = col_cutter.getColumns()
                    if not self.isLegalCol(cols, type):
                        continue
                    ori, gray = col_cutter.getImg()
                    gray_shape = gray.shape
                    if gray_shape[0] != 790:
                        continue
                    img_name = col_cutter.getImgName()
                    text_cutter = V_TextCutter(gray, ori, self.Info, cols, text, type, img_name, double_lines,
                                               pic_words, ori_t, img_with_line)
                    text_cutter.cutColumns()
                    self.tot += 1
                except:
                    t = re.findall(r'四库全书.*\.png', img_name)
                    continue
            except:
                print ("AN ERROR OCCURS!")
                continue
        self.Info.save()

    def isLegalCol(self, cols, type):
        if type == 'b':
            l_bond = 0
            r_bond = 1
        else:
            l_bond = 1
            r_bond = 0
            # 根据类型决定删掉哪行
        for i in range(l_bond, len(cols) - r_bond):
            p = cols[i]
            if p[1] - p[0] < 60 or p[1] - p[0] > 80:
                return False
        return True

    def makeData_MT(self):
        p1 = Producer(self.file_list, self.Info)
        p1.start()
        consumers = [Consumer() for i in range(32)]
        for c in consumers:
            c.start()
        for c in consumers:
            c.join()
        self.Info.save()
