# -*- coding:utf-8 -*-
# @Author: Wei Yi

import cv2
import numpy as np
from Information import Information
import math
import time
import os
import re


class V_ColumnCutter:

    def __init__(self, file, type):
        self.gray = cv2.imdecode(np.fromfile(file, dtype=np.uint8), -1)
        if len(self.gray.shape) == 3:
            self.gray = cv2.cvtColor(self.gray, cv2.COLOR_BGR2GRAY)
        self.img = np.copy(self.gray)
        eraser = Eraser(self.img)
        eraser.eraserLine_CV()
        self.ori = np.copy(self.img)
        self.img = cv2.cvtColor(self.img, cv2.COLOR_GRAY2BGR)
        self.shape = self.gray.shape
        self.columns = []
        self.img_name = file
        self.shape = self.gray.shape
        self.A = None
        self.B = None
        self.delta = None
        self.psi = None
        if type == 'a':
            self.text = "DAAAAAAAA"
        else:
            self.text = "AAAAAAAAD"
        self.tot = len(self.text)
        self.col, self.points = self.getPoints()
        self.initVar()

    def toClear(self):
        # 是图片模糊化（让文字尽可能连在一起）然后通过原图加上文本分割的竖线
        # 参数self.gray = cv2.GaussianBlur(self.gray, (7, 7), 2)
        self.gray = cv2.GaussianBlur(self.gray, (7, 7), 2)
        img_shape = self.shape
        for i in range(img_shape[0]):
            for j in range(img_shape[1]):
                if self.gray[i, j] < 180:
                    self.gray[i, j] = 0
                else:
                    self.gray[i, j] = 255
        for i in range(img_shape[0]):
            for j in range(img_shape[1]):
                if self.ori[i, j] == 0 and self.gray[i, j] != 0:
                    self.gray[i, j] == 0

    def getColumns(self):
        for i in range(len(self.points)):
            self.delta[i][0] = self.fun_Pi(self.points[i]) + self.B[i]
            self.psi[i][0] = i
        for i in range(1, self.tot + 1):
            for j in range(len(self.points)):
                pos = -1
                for k in range(len(self.points)):
                    a = 0
                    if self.text[i - 1] != "D":
                        a = self.A[k][j][0]
                    else:
                        a = self.A[k][j][1]
                    gain = a + self.B[j]
                    if self.delta[k][i - 1] + gain > self.delta[j][i]:
                        pos = k
                        self.delta[j][i] = self.delta[k][i - 1] + gain
                self.psi[j][i] = pos
        pos = -1
        max = -1
        for i in range(len(self.points)):
            if self.delta[i][self.tot] >= max:
                max = self.delta[i][self.tot]
                pos = i
        trace = self.backTrace(pos)
        st = 0
        fin = self.shape[0]
        # print (trace)
        for i in range(self.tot, -1, -1):
            marL = 0
            marR = 0
            if i != 0:
                marL = self.findWhite(self.points[trace[i]], -1)
                marR = self.findWhite(self.points[trace[i]], 1)
                self.columns.append((self.points[trace[i]] + marL, self.points[trace[i - 1]] + marR))
            y = self.points[trace[i]] + marL
            cv2.line(self.img, (y, st), (y, fin), (255, 0, 0), 1)
        return self.columns, self.img

    def findWhite(self, s, d):
        cnt = 0
        tot_w = self.shape[0]
        while abs(cnt) < 10:
            if self.col[s + cnt] != tot_w:
                cnt += d
            else:
                break
        if abs(cnt) < 10:
            return cnt
        else:
            return 0

    def backTrace(self, pos):
        trace = []
        trace.append(pos)
        st = pos
        for i in range(self.tot, 0, -1):
            st = self.psi[st][i]
            trace.append(st)
        return trace

    def fun_Pi(self, pos):
        if pos == 1 or pos == 0:
            return 0
        return (1 - pos / self.shape[1]) * self.shape[1]

    def fun_A(self, length):
        if length <= 0:
            return (length - 100, length - 100)
        r1 = 0
        if length <= 50 and length >= 70:
            r1 = length / 45
        elif length >= 70 and length <= 80:
            temp = (length - 55) // 10
            r1 = 1 - (temp / 100)
        elif length >= 80 or length < 50:
            r1 = -1
        else:
            r1 = 1
        v1 = 60 * r1
        r2 = 0
        if length <= 30:
            r2 = length / 30
        elif length >= 40 and length <= 75:
            temp = (length - 40) // 10
            r2 = 1 - (temp / 100)
        elif length >= 75:
            r2 = -1
        else:
            r2 = 1
        if length < 15:
            r2 = -1
        v2 = 40 * r1
        if v1 > 0:
            v1 = math.log(v1)
        if v2 > 0:
            v2 = math.log(v2)
        return (v1, v2)

    def fun_B(self, pix):
        if pix == 0:
            return 0
        return math.log(pix * 255)

    def getPoints(self):
        col = []
        for i in range(self.shape[1]):
            tot = 0
            for j in range(self.shape[0]):
                if self.gray[j, i] == 0:
                    tot += 1
            col.append(tot)
        points = [1]
        # print (col)
        for i in range(1, self.shape[1]):
            if col[i - 1] - col[i] >= 50:
                points.append(i - 1)
            if col[i] - col[i - 1] >= 50:
                points.append(i)
        points.append(self.shape[1] - 1)
        return col, points

    def initVar(self):
        l = len(self.points)
        self.A = [[[0, 0] for i in range(l)] for j in range(l)]
        for i in range(l):
            for j in range(l):
                self.A[i][j] = self.fun_A(self.points[j] - self.points[i])
        self.B = [0 for i in range(l)]
        for i in range(l):
            self.B[i] = self.fun_B(self.col[self.points[i]])
        self.delta = [[0 for i in range(self.tot + 1)] for j in range(l)]
        self.psi = [[0 for i in range(self.tot + 1)] for j in range(l)]

    def getImg(self):
        eraser = Eraser(self.gray)
        eraser.eraserLine_CV()
        return self.ori, self.gray

    def getImgName(self):
        return self.img_name


class V_TextCutter:
    def __init__(self, gray, ori, info, columns, text, type, img_name, double_lines, pic_word, ori_text, img_with_line):
        """
        :param gray: 模糊后的图片，减小了每个文字内的空隙
        :param ori: 原始图片，用于切分
        :param info: 类，用于获得 字-序号 哈希表，已处理的图片名集合，写日志
        :param columns: 列元组的列表
        :param text: 文本列表，n*m，n为列数，m为每行的文字数（用D代替双行内容）
        :param type:a或b，a删除最左行，b删除最后行
        :param img_name:文件名，用于加入info类的processed_img集合
        :param double_lines:双行内容，n*2m，n为双行数，m为每个双行的但行数。m中i为右边的字，i+1为左边的字（空白用N代替）
        """
        self.gray = gray
        self.ori = ori
        self.img_with_line = img_with_line
        self.shape = self.gray.shape
        self.info = info
        self.columns = columns
        self.tot_text = text
        self.double_lines = double_lines
        self.tot = 0
        self.cur_col = 0
        self.type = type
        self.img_name = img_name
        self.dl = 0
        self.dc = 0
        self.picword = 0
        self.pic_word = pic_word
        self.ori_text = ori_text
        t = re.findall(r'四库全书.*\.png', self.img_name)
        self.img_file_name = t[0][:-4]

    def decode(self, st, fin):
        for i in range(len(self.points)):
            self.delta[i][0] = self.fun_Pi(self.points[i]) + self.B[i]
            self.psi[i][0] = i
        for i in range(1, self.tot + 1):
            for j in range(len(self.points)):
                pos = -1
                for k in range(0, j):
                    a = 0
                    if self.text[i - 1] != "D":
                        a = self.A[k][j][0]
                    else:
                        a = self.A[k][j][1]
                    gain = a + self.B[j]
                    if self.delta[k][i - 1] + gain >= self.delta[j][i]:
                        if self.col != self.tot and self.delta[k][i - 1] + gain == self.delta[j][i]:
                            continue
                        pos = k
                        self.delta[j][i] = self.delta[k][i - 1] + gain
                self.psi[j][i] = pos
        pos = -1
        max = -1
        for i in range(len(self.points)):
            if self.delta[i][self.tot] >= max:
                max = self.delta[i][self.tot]
                pos = i
        trace = self.backTrace(pos)
        word_pos = []
        if min(trace) == -1:
            return word_pos
        for i in range(self.tot, -1, -1):
            marU = 0
            marD = 0
            if i != 0:
                marU = self.findWhite(self.points[trace[i]], -1)
                """
                if self.tot - i != 0:
                    if self.tot_text[self.cur_col][self.tot - i -1] == '一' or self.tot_text[self.cur_col][self.tot - i -1] == '二' or self.tot_text[self.cur_col][self.tot - i -1] == '三':
                        mar = 15
                """
                marD = self.findWhite(self.points[trace[i]], 1)
                word_pos.append((self.points[trace[i]] + marU, self.points[trace[i - 1]] + marD))

            y = self.points[trace[i]] + marU
            cv2.line(self.img_with_line, (st, y), (fin, y), (0, 0, 255), 1)
        return word_pos

    def findWhite(self, s, d):
        cnt = 0
        while abs(cnt) < 10:
            if self.col[s + cnt] != self.tot_w:
                cnt += d
            else:
                break
        if abs(cnt) < 10:
            return cnt
        else:
            return 0

    def backTrace(self, pos):
        trace = []
        trace.append(pos)
        st = pos
        for i in range(self.tot, 0, -1):
            st = self.psi[st][i]
            trace.append(st)
        return trace

    def fun_Pi(self, pos):
        if pos == 1 or pos == 0:
            return 0
        return math.log(pos, 0.5)

    def fun_A(self, length):
        if length <= 0:
            length = -1000000
            return (length - 100, length - 100)
        r1 = 0
        if length <= 45 and length >= 30:
            r1 = length / 45
        elif length >= 55 and length <= 80:
            temp = (length - 55) // 10
            r1 = 1 - (temp / 100)
        elif length >= 80 or length <= 30:
            r1 = -1
        else:
            r1 = 1
        if length < 30:
            r1 = -1
        v1 = 50 * r1
        r2 = 0
        if length <= 30 and length >= 20:
            r2 = length / 30
        elif length >= 40 and length <= 60:
            temp = (length - 40) // 10
            r2 = 1 - (temp / 100)
        elif length >= 60 or length <= 20:
            r2 = -1
        else:
            r2 = 1
        if length < 15:
            r2 = -1
        v2 = 35 * r1
        if v1 > 0:
            v1 = math.log(v1)
        if v2 > 0:
            v2 = math.log(v2)
        return (v1, v2)

    def fun_B(self, pix):
        if pix == 0:
            return 0
        return pix

    def getPoints(self, st, fin):
        col = []
        for i in range(self.shape[0]):
            tot = 0
            for j in range(st, fin):
                if self.gray[i, j] == 255:
                    tot += 1
            col.append(tot)
        points = []
        flag = False
        tot = fin - st
        self.tot_w = tot
        for i in range(1, len(col)):
            if not flag and col[i - 1] - col[i] >= 3:
                flag = True
                if i > 5:
                    points.append(i - 5)
                else:
                    points.append(i)
            if flag and (col[i] - col[i - 20] >= 3) and tot - col[i] <= 10:
                points.append(i)

        return col, points

    def truncatePoins(self, points, dis=1):
        poi = []
        st = points[0]
        fin = -1
        for i in range(1, len(points)):
            if points[i] != st + dis:
                if fin == -1:
                    poi.append(st)
                else:
                    poi.append((fin + st) // 2)
                st = points[i]
                fin = -1
                continue
            fin = points[i]
        poi.append(points[-1])
        return poi

    def initVar(self, text, st, fin):
        self.A = None
        self.B = None
        self.delta = None
        self.psi = None
        self.text = text
        self.tot = len(self.text)
        self.col, self.points = self.getPoints(st, fin)
        l = len(self.points)
        self.A = [[[0, 0] for i in range(l)] for j in range(l)]
        for i in range(l):
            for j in range(l):
                self.A[i][j] = self.fun_A(self.points[j] - self.points[i])
        self.B = [0 for i in range(l)]
        for i in range(l):
            self.B[i] = self.fun_B(self.col[self.points[i]])
        self.delta = [[0 for i in range(self.tot + 1)] for j in range(l)]
        self.psi = [[-100 for i in range(self.tot + 1)] for j in range(l)]

    def cutColumns(self):
        if self.img_name in self.info.processed_img:
            print ("This image has already been processed!")
            return
        if self.type == 'b':
            l_bond = 2
            r_bond = -1
        else:
            l_bond = 1
            r_bond = 0
        # 根据类型决定删掉哪行
        col_only = False
        for i in range(len(self.columns) - l_bond, r_bond, -1):
            if self.cur_col == len(self.tot_text):
                break
            if (self.tot_text[self.cur_col] == ''):
                self.cur_col += 1
                continue
            self.text_pos = []
            self.dc = 0
            pair = self.columns[i]
            st = pair[0]
            fin = pair[1]
            self.initVar(self.tot_text[self.cur_col], st, fin)
            word_pos = self.cutCol(st, fin)
            if word_pos != [] and self.isLegalLength(self.points[0], self.points[-1], word_pos):
                self.saveCharaters(st, fin, word_pos)
                self.saveCol(st, fin, word_pos)
            else:
                if word_pos != []:
                    self.saveCol(st, fin, word_pos)
                self.dealUncut()
            self.cur_col += 1
        pos = self.img_file_name.find("四库全书_")
        path = "RES\\" + self.img_file_name[pos:] + "_cut.png"
        cv2.imencode('.png', self.img_with_line)[1].tofile(path)

    def saveCol(self, st, fin, word_pos):
        text = []
        temp = ""
        for ch in self.ori_text[self.cur_col]:
            if ch == '[':
                if temp != "":
                    text.append(temp)
                    temp = ""
                continue
            if ch == ']':
                text.append(temp)
                temp = ""
                continue
            temp += ch
        text.append(temp)
        s = word_pos[0][0]
        f = word_pos[0][1]
        col_text = self.tot_text[self.cur_col]
        cnt = 0
        for p in range(1, len(col_text)):
            if col_text[p - 1] == 'D' and col_text[p] != 'D':
                name = text[cnt].replace('/', '~')
                if (not (os.path.exists("COL"))):
                    os.mkdir("COL")
                if (not (os.path.exists("COL\\" + self.info.save_path))):
                    os.mkdir("COL\\" + self.info.save_path)
                pos = self.img_file_name.find("四库全书_")
                path = "COL\\" + self.info.save_path + "\\" + self.img_file_name[pos:] + '-' + name + ".png"
                cv2.imencode('.png', self.ori[s:f, st:fin])[1].tofile(path)
                cnt += 1
                s = f
                f = word_pos[p][1]
            elif col_text[p - 1] != 'D' and col_text[p] == 'D':
                name = text[cnt].replace('/', '~')
                if (not (os.path.exists("COL"))):
                    os.mkdir("COL")
                if (not (os.path.exists("COL\\" + self.info.save_path))):
                    os.mkdir("COL\\" + self.info.save_path)
                pos = self.img_file_name.find("四库全书_")
                path = "COL\\" + self.info.save_path + "\\" + self.img_file_name[pos:] + '-' + name + ".png"
                cv2.imencode('.png', self.ori[s:f, st:fin])[1].tofile(path)
                cnt += 1
                s = f
                f = word_pos[p][1]
            else:
                f = word_pos[p][1]
        name = text[cnt].replace('/', '~')
        if (not (os.path.exists("COL"))):
            os.mkdir("COL")
        if (not (os.path.exists("COL\\" + self.info.save_path))):
            os.mkdir("COL\\" + self.info.save_path)
        pos = self.img_file_name.find("四库全书_")
        path = "COL\\" + self.info.save_path + "\\" + self.img_file_name[pos:] + '-' + name + ".png"
        cv2.imencode('.png', self.ori[s:f, st:fin])[1].tofile(path)
        cnt += 1

    def isLegalLength(self, st, fin, word_pos):
        tot = 0
        for c in self.tot_text[self.cur_col]:
            tot += 36
        if (fin - st) - tot >= 37 or (fin - st) - tot <= 0:
            return False
        for i in range(len(word_pos) - 1):
            if self.col[word_pos[i][0]] != self.tot_w or self.col[word_pos[i][1]] != self.tot_w:
                return False
            if word_pos[i][1] - word_pos[i][0] > 60:
                return False
            if word_pos[i][1] - word_pos[i][0] < 10:
                return False
        return True

    def saveCharaters(self, st, fin, word_pos):
        cnt = 0
        pos = -1
        for c in self.tot_text[self.cur_col]:
            pos += 1
            if c != 'D':
                # 处理单行
                if pos != 0 and self.tot_text[self.cur_col][pos - 1] == 'D':
                    self.dl += 1
                    self.dc = 0
                if c == 'P':
                    c = self.pic_word[self.picword]
                    self.picword += 1
                self.save(c, st, fin, word_pos[cnt])
            else:
                # 处理双行，c1为右边的字，c2为左边的字（为N则跳过）
                c1 = self.double_lines[self.dl][self.dc]
                if c1 != 'N':
                    if c1 == 'P':
                        c1 = self.pic_word[self.picword]
                        self.picword += 1
                    self.save(c1, st, fin, word_pos[cnt], 'second')
                if self.dc + 1 != len(self.double_lines[self.dl]):
                    c2 = self.double_lines[self.dl][self.dc + 1]
                    self.dc += 2
                    if c2 != 'N':
                        if c2 == 'P':
                            c2 = self.pic_word[self.picword]
                            self.picword += 1
                        self.save(c2, st, fin, word_pos[cnt], 'first')
                if pos == len(self.tot_text[self.cur_col]) - 1:
                    self.dl += 1
                    self.dc = 0
            cnt += 1

    def save(self, c, st, fin, word_pos, part=None):
        """
        :param c: 字符
        :param st: 列坐标
        :param fin: 列坐标
        :param cnt: 当前元组序号（字的坐标）
        :param part: 双行字，first为左字
        参数：resized = cv2.resize(temp, (50, 50), interpolation=cv2.INTER_AREA)
        """
        if not self.info.hasCharater(c):
            # 检查是否已有此字
            self.info.addCharacter(c)
        s = word_pos[0]
        f = word_pos[1]
        if part != None:
            # 切双行字，二分
            mid = int((fin - st) / 2)
            if part == "first":
                fin = st + mid
            else:
                st = st + mid
        path = self.info.character_map[c][0]
        ch_ord = self.info.character_map[c][1]
        ord = self.info.character_map[c][2]
        pos = self.img_file_name.find("四库全书_")
        path = path + "\\" + self.img_file_name[pos:] + '-' + c + '-' + str(ord) + '.png'
        # print(path)
        # 路径名，例..\\train\\1\\1-10
        try:
            if f - s > 10:
                temp = self.ori[s:f, st:fin]
                # resized = cv2.resize(temp, (64, 64), interpolation=cv2.INTER_AREA)
                # cv2.imwrite(path, resized)
                cv2.imencode('.png', temp)[1].tofile(path)
                if self.info.character_map[c][2] <= 10000:
                    self.info.character_map[c][2] += 1
                self.info.processed_img.add(self.img_name)
        except:
            pass

    def dealUncut(self):
        pos = -1
        # cv2.imshow('img-', self.img_with_line)
        # cv2.waitKey(0)
        for c in self.tot_text[self.cur_col]:
            pos += 1
            if c != 'D':
                # 处理单行
                if pos != 0 and self.tot_text[self.cur_col][pos - 1] == 'D':
                    self.dl += 1
                    self.dc = 0
                if c == 'P':
                    self.picword += 1
            else:
                # 处理双行，c1为右边的字，c2为左边的字（为N则跳过）
                c1 = self.double_lines[self.dl][self.dc]
                if c1 == 'P':
                    self.picword += 1
                if self.dc + 1 != len(self.double_lines[self.dl]):
                    c2 = self.double_lines[self.dl][self.dc + 1]
                    self.dc += 2
                    if c2 != 'N':
                        if c2 == 'P':
                            self.picword += 1
                if pos == len(self.tot_text[self.cur_col]) - 1:
                    self.dl += 1
                    self.dc = 0

    def cutCol(self, st, fin):
        return self.decode(st, fin)


class Eraser:
    def __init__(self, img):
        self.gray = img
        self.shape = self.gray.shape

    def eraserLine(self):
        pass

    def eraserLine_CV(self):
        low_threshold = 50
        high_threshold = 150
        edges = cv2.Canny(self.gray, low_threshold, high_threshold)
        rho = 1  # distance resolution in pixels of the Hough grid
        theta = np.pi / 180  # angular resolution in radians of the Hough grid
        threshold = 50  # minimum number of votes (intersections in Hough grid cell)
        min_line_length = 50  # minimum number of pixels making up a line
        max_line_gap = 1  # maximum gap in pixels between connectable line segments
        # line_image = np.copy(self.gray) * 0  # creating a blank to draw lines on

        # Run Hough on edge detected image
        # Output "lines" is an array containing endpoints of detected line segments
        lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                                min_line_length, max_line_gap)
        for line in lines:
            for x1, y1, x2, y2 in line:
                if x1 == x2 and y1 - y2 >= 50:
                    cv2.line(self.gray, (x1 - 2, 0), (x2 - 2, self.shape[0] - 1), (255, 255, 255), 1)
                    cv2.line(self.gray, (x1, 0), (x2, self.shape[0] - 1), (255, 255, 255), 5)
                    cv2.line(self.gray, (x1 + 2, 0), (x2 + 2, self.shape[0] - 1), (255, 255, 255), 1)
        cv2.line(self.gray, (0, 0), (0, self.shape[0] - 1), (255, 255, 255), 3)
        cv2.line(self.gray, (self.shape[1] - 1, 0), (self.shape[1] - 1, self.shape[0] - 1), (255, 255, 255), 3)

        low_threshold = 50
        high_threshold = 150
        edges = cv2.Canny(self.gray, low_threshold, high_threshold)
        rho = 1  # distance resolution in pixels of the Hough grid
        theta = np.pi / 180  # angular resolution in radians of the Hough grid
        threshold = 50  # minimum number of votes (intersections in Hough grid cell)
        min_line_length = 50  # minimum number of pixels making up a line
        max_line_gap = 50  # maximum gap in pixels between connectable line segments
        # line_image = np.copy(self.gray) * 0  # creating a blank to draw lines on

        # Run Hough on edge detected image
        # Output "lines" is an array containing endpoints of detected line segments
        lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                                min_line_length, max_line_gap)
        for line in lines:
            for x1, y1, x2, y2 in line:
                if y1 == y2:
                    if y1 > 50 and y1 < self.shape[0] - 50:
                        continue
                    cv2.line(self.gray, (0, y1), (self.shape[1] - 1, y2), (255, 255, 255), 5)
        cv2.line(self.gray, (0, 5), (self.shape[1] - 1, 5), (255, 255, 255), 10)
        cv2.line(self.gray, (0, self.shape[0] - 5), (self.shape[1] - 1, self.shape[0] - 5), (255, 255, 255), 10)
        return self.gray
