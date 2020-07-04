# -*- coding:utf-8 -*-
# @Author: Wei Yi

import os
from MakeData import MakeData
import random


class Img2CharactersImg:
    def __init__(self, path, book_name, load=False):
        self.load = load
        self.file_names = []
        self.path = path
        files = os.listdir(path)
        self.book_name = book_name
        self.getFileList(path + "\\", files)
        self.makeData()

    def getFileList(self, path, files):
        for f in files:
            f_full = path + f
            if os.path.isdir(f_full):
                folders = os.listdir(f_full)
                cur_path = f_full + "\\"
                self.getFileList(cur_path, folders)
            else:
                if f.find(".png") != -1:
                    self.file_names.append(path + f[:-4])

    def makeData(self):
        data_maker = MakeData(self.file_names, self.book_name, self.load)
        # data_maker.makeData_MT()
        data_maker.makeData()
