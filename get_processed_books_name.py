# -*- coding:utf-8 -*-
# @Author: Wei Yi

import os
import pickle
from collections import defaultdict

path = "单字\\"
books = os.listdir(path)
processed_books = defaultdict(int)
for b in books:
    b = b.replace("-w", '')
    processed_books[b] = 1
print(len(processed_books))
output = open('Processed_Books.pkl', 'wb')
pickle.dump(processed_books, output)
output.close()
