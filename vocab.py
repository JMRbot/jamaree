from pythainlp.tokenize import word_tokenize
import deepcut
import pandas as pd
import h5py
import itertools

all_doc = []
dataX = []
dataY = []
data=open("conv2000plus.txt","r").read()
for line in data.split("\n\n"):
    wordQA = line.split("\n")
    question=wordQA[0]
    answer=wordQA[1]
    dataX.append(question)
    dataY.append(answer)
line = dataX+dataY

cumsub=[]
sumAll =  []
for vocab in line:
    cumsub = deepcut.tokenize(vocab)
    for w in cumsub:
        sumAll.append(w)
print('cumsub',sumAll)
wordset = set(sumAll)
print('wordset',wordset)

##เขียนไฟล์
f = open('vocab.txt','w+') #เปิดไฟล์ vocab.txt ถ้ายังไม่มีไฟล์ ให้สร้างไฟล์ใหม่
for vocablist in wordset:
    f.write(vocablist+"\n") #เก็บค่า vocab ลงไป
f.close() #ปิดไฟล์




