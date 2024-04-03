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

def prepare_labeled_data():
    # 输入和输出文件的路径
    input_file_path = 'file/DSTC2_csv/train_data.csv'
    output_file_path = 'file/DSTC2_csv/train_data_before_label.csv'
    output_after = 'file/DSTC2_csv/no_labeled_data.csv'
    output_file1 = 'file/DSTC2_csv/train_data_after_semi.csv'

    # 读取输入文件
    with open(input_file_path, 'r', encoding='utf-8') as input_file:
        reader = csv.reader(input_file)

        # 将所有行读入一个列表中
        rows = list(reader)

    # 计算应保留的行数（前一半）
    half_row_count = len(rows) // 2

    # 保存前一半的内容到输出文件
    with open(output_file_path, 'w', encoding='utf-8', newline='') as output_file:
        writer = csv.writer(output_file)

        # 写入前一半的行
        writer.writerows(rows[:half_row_count])
    with open(output_file1, 'w', encoding='utf-8', newline='') as output_file_new:
        writer = csv.writer(output_file_new)

        # 写入前一半的行
        writer.writerows(rows[:half_row_count])

    # 保存后一半的内容到另一个输出文件
    with open(output_after, 'w', encoding='utf-8', newline='') as output_file_after:
        writer = csv.writer(output_file_after)

        # 写入后一半的行
        writer.writerows(rows[half_row_count:])



if __name__ == '__main__':
    # trainDataToCsv()
    # testDataToCsv()
    prepare_labeled_data()