# 在rasa/data/file中，遍历全部的label（用户说的话）
import csv
import json
import pandas as pd
import os
from sklearn.utils import shuffle
import re
import sys


def findLabelFile(base):
    for root, ds, fs in os.walk(base):
        for f in fs:
            fullname = os.path.join(root, f)
            if fullname.__contains__("label"):
                yield fullname


# 在rasa/data/file中，遍历全部的log（model回复的话）
def findLogFile(base):
    for root, ds, fs in os.walk(base):
        for f in fs:
            fullname = os.path.join(root, f)
            if fullname.__contains__("log"):
                yield fullname


def trainDataToCsv():
    file = open("file/DSTC2_csv/train_data.csv", "w")
    sys.stdout = file
    base = 'file'
    labels = []
    logs = []
    for i in findLabelFile(base):
        labels.append(i)
    for i in findLogFile(base):
        logs.append(i)
    dictIntent = {'inform': [], 'request': [], 'thankyou': [], 'repeat': [], 'reqalts': [], 'affirm': [],
                  'negate': [],
                  'hello': [], 'bye': [], 'restart': [], 'confirm': [], 'ack': [], 'deny': [], 'null()': []}
    intent_matches = dictIntent.keys()
    # print(intent_matches)

    for i in range(0, len(labels)):
        labelText = open(labels[i]).read()
        labelText = json.loads(labelText)

        for turn in labelText["turns"]:
            userWord = turn.get("transcription", "")
            meanOfUser = turn.get("semantics", "").get("cam")

            # print(userWord)
            # print(meanOfUser)
            judge = False
            for intent in intent_matches:
                if intent in meanOfUser or meanOfUser == 'reqmore()':
                    judge = True
                    if meanOfUser == 'reqmore()':
                        intent = 'reqalts'

                    dictIntent[intent].append(userWord)
                    break
            if not judge:
                print(meanOfUser)

    for key in dictIntent.keys():
        temp = dictIntent[key]
        for e in temp:
            tmp = key + ", " + e
            print(tmp)
    file.close()
    csvFileRandom("file/DSTC2_csv/train_data.csv")


def testDataToCsv():
    file = open("file/DSTC2_csv/test_data.csv", "w")
    sys.stdout = file
    base = 'file/test'
    labels = []
    logs = []
    for i in findLabelFile(base):
        labels.append(i)
    for i in findLogFile(base):
        logs.append(i)
    dictIntent = {'inform': [], 'request': [], 'thankyou': [], 'repeat': [], 'reqalts': [], 'affirm': [],
                  'negate': [],
                  'hello': [], 'bye': [], 'restart': [], 'confirm': [], 'ack': [], 'deny': [], 'null()': []}
    intent_matches = dictIntent.keys()
    # print(intent_matches)

    for i in range(0, len(labels)):
        labelText = open(labels[i]).read()
        labelText = json.loads(labelText)

        for turn in labelText["turns"]:
            userWord = turn.get("transcription", "")
            meanOfUser = turn.get("semantics", "").get("cam")

            # print(userWord)
            # print(meanOfUser)
            judge = False
            for intent in intent_matches:
                if intent in meanOfUser or meanOfUser == 'reqmore()':
                    judge = True
                    if meanOfUser == 'reqmore()':
                        intent = 'reqalts'

                    dictIntent[intent].append(userWord)
                    break
            if not judge:
                print(meanOfUser)

    for key in dictIntent.keys():
        temp = dictIntent[key]
        for e in temp:
            tmp = key + ", " + e
            print(tmp)
    file.close()
    csvFileRandom("file/DSTC2_csv/test_data.csv")


def csvFileRandom(file_path):
    data = pd.read_csv(file_path)
    data = shuffle(data)
    data.to_csv(file_path)
    modified_content = []
    with open(file_path, mode='r', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            # 去除第一个元素并保存修改后的行
            modified_content.append(row[1:])
    with open(file_path, mode='w', encoding='utf-8', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        for row in modified_content:
            csv_writer.writerow(row)






if __name__ == '__main__':
    # trainDataToCsv()
    testDataToCsv()
