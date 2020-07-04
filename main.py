# -*- coding:utf-8 -*-
# @Author: Wei Yi

from Img2CharactersImg import Img2CharactersImg
import time
import os
from collections import defaultdict
import pickle
import random

LOAD = False
path = "./Test"
books = os.listdir(path)
if not os.path.exists("Res"): os.mkdir("Res")
if LOAD:
    print("Loading...")
    pkl_file = open('Processed_Books.pkl', 'rb')
    processed_books = pickle.load(pkl_file)
    pkl_file.close()
else:
    processed_books = defaultdict(int)
cnt = 0
for f in books:
    if cnt == 100:
        break
    try:
        if processed_books[f] == 0:
            print(f)
            cnt += 1
            processed_books[f] += 1
            output = open('Processed_Books.pkl', 'wb')
            pickle.dump(processed_books, output)
            output.close()
            cur_path = path + "\\" + f
            i2c = Img2CharactersImg(cur_path, f, load=False)
        else:
            continue
    except:
        output = open('Processed_Books.pkl', 'wb')
        pickle.dump(processed_books, output)
        output.close()
        continue
output = open('Processed_Books.pkl', 'wb')
pickle.dump(processed_books, output)
output.close()
