# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 15:39:10 2020

@author: Alex
"""

import os

test_dir = 'C:/Users/Alex/Documents/School/AI/AI-Class-Project/test/test/'
filenames = os.listdir(test_dir)
for file in filenames:
    offset = ""
    if len(file) < 9:
        offset = "0" * (9-len(file))
    #print(test_dir + offset + file)
    os.rename(test_dir + file, test_dir + offset + file)