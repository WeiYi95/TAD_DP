import re


class TextProcessor:

    def __init__(self, file):
        self.file = open(file, 'r', encoding="utf-8")
        self.file_name = file
        self.ori = None

    def getType(self):
        try:
            pos = self.file_name.find(']')
            type = self.file_name[pos - 1]
        except:
            type = 'n'
            return type
        return type

    def getText(self):
        content = self.file.read()
        content = content.split('\n')
        content = content[0]
        self.ori = content.split('-')
        content = content.split('-')
        return content

    def processText(self, content):
        double_lines = []
        pic_words = []
        text = []
        for c in content:
            c = c.strip()
            c = c.replace('　', '')
            tempp = re.findall(r'{.*?}', c)
            if tempp != []:
                for sth in tempp:
                    pic_words.append(sth)
                c = re.sub(r'{.*?}', 'P', c)
            tempd = re.findall(r'\[.*?\]', c)
            lens = []
            if tempd != []:
                for sth in tempd:
                    line = sth[1:-1]
                    pos = line.find('/')
                    l = line[:pos]
                    r = line[pos + 1:]
                    length = max(len(l), len(r))
                    temp_list = []
                    for i in range(length):
                        if i >= len(l):
                            temp_list.append('N')
                        else:
                            temp_list.append(l[i])
                        if i >= len(r):
                            temp_list.append('N')
                        else:
                            temp_list.append(r[i])
                    double_lines.append(temp_list)
                    lens.append(length)
            c = self.replaceDoubleLines(lens, c)
            text.append(c)
        return text, double_lines, pic_words

    def replaceDoubleLines(self, lens, c):
        for l in lens:
            d = 'D' * l
            lpos = c.find('[')
            rpos = c.find(']')
            c = c[:lpos] + d + c[rpos + 1:]
        return c

    def getProcessText(self):
        content = self.getText()
        text, double_lines, pic_words = self.processText(content)
        self.file.close()
        return text, double_lines, pic_words, self.ori
