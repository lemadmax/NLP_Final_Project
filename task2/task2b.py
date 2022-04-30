from ast import Mod
from operator import mod
from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from nltk.stem import PorterStemmer
import numpy as np
import math
import sys

############# Utils Start ################

partitives = dict()

features = ["Token", "POS", "BIO", "Num", "Dis2Sup", "Dis2Pred"]

## Generate sklearn usable data and target from data file.
def GenerateDataFromFile(filename):
    file = open(filename, 'r', encoding='UTF-8')
    lines = file.readlines()
    file.close()
    # sample_cnt = len(lines)
    # print(sample_cnt)
    data_x = []
    data_y = []

    start_index = 0
    sup_index = -1
    pred_index = -1
    data_cnt = 0
    for index in range(len(lines)):
        fields = lines[index].split()
        if len(fields) == 0:
            ## Add distance to support and distance to pred features
            for data_index in range(start_index, data_cnt):
                dis2sup = abs(data_index - sup_index)
                if sup_index == -1:
                    dis2sup = 50
                data_x[data_index].append(dis2sup)
                data_x[data_index].append(abs(data_index - pred_index))
            sup_index = -1
            pred_index = -1
            continue
        if fields[3] == "0":
            start_index = data_cnt
        ## Add POS, BIO, NUM features
        features = fields[0:4]

        data_x.append(features)
        data_y.append(0)
        if len(fields) > 5:
            if fields[5] == "SUPPORT":
                sup_index = data_cnt
            elif fields[5] == "PRED":
                pred_index = data_cnt
            elif fields[5] == "ARG1": ## Set target y value to 2(represent ARG1, 0 for no arg)
                data_y[data_cnt] = 1
        data_cnt += 1
    # print(data_x[:25])
    # print(data_y)
    return data_x, data_y

def GenerateDataFromFile_t2b(filename):
    file = open(filename, 'r', encoding='UTF-8')
    lines = file.readlines()
    file.close()
    # sample_cnt = len(lines)
    # print(sample_cnt)
    data_x = []
    data_y = []

    ps = PorterStemmer()

    start_index = 0
    pred_index = -1
    sup_index = -1
    cc_index = -1
    data_cnt = 0
    type_map = dict()
    for index in range(len(lines)):
        fields = lines[index].split()
        if len(fields) == 0:
            for i in range(start_index, data_cnt):

                ## previous two words
                pre_near_begin = False
                pre_index = i - 1
                if pre_index < start_index:
                    data_x[i].extend(["", "", ""])
                    pre_near_begin = True
                else:
                    data_x[i].extend(data_x[pre_index][0:3])
                pre_index -= 1
                if pre_index < start_index:
                    data_x[i].extend(["", "", ""])
                    pre_near_begin = True
                else:
                    data_x[i].extend(data_x[pre_index][0:3])
                if pre_near_begin:
                    data_x[i].append(1)
                else:
                    data_x[i].append(0)
                
                ## two words after
                fol_near_end = False
                fol_index = i + 1
                if fol_index >= data_cnt:
                    data_x[i].extend(["", "", ""])
                    fol_near_end = True
                else:
                    data_x[i].extend(data_x[fol_index][0:3])
                fol_index += 1
                if fol_index >= data_cnt:
                    data_x[i].extend(["", "", ""])
                    fol_near_end = True
                else:
                    data_x[i].extend(data_x[fol_index][0:3])
                if fol_near_end:
                    data_x[i].append(1)
                else:
                    data_x[i].append(0)
                if i + 1 < data_cnt and data_x[i][1][0] == 'N' and data_x[i+1][2][0] != 'I':
                    data_x[i].append(1)
                else:
                    data_x[i].append(0)

                ## distance from pred, sup, and cc
                dis2pred = [0, 0, 0, 0, 0, 0]
                dis2sup = [0, 0, 0, 0, 0, 0]
                dis2cc = [0, 0, 0, 0, 0, 0]
                pre_index = i - 1
                if pre_index >= start_index:
                    if pre_index in type_map:
                        if type_map[pre_index] == "PRED":
                            dis2pred[2] = 1
                            dis2pred[1] = 1
                            dis2pred[0] = 1
                        elif type_map[pre_index] == "SUPPORT":
                            dis2sup[2] = 1
                            dis2sup[1] = 1
                            dis2sup[0] = 1
                    if data_x[pre_index][1] == "CC":
                        dis2cc[2] = 1
                        dis2cc[1] = 1
                        dis2cc[0] = 1
                pre_index = i - 2
                if pre_index >= start_index:
                    if pre_index in type_map:
                        if type_map[pre_index] == "PRED":
                            dis2pred[1] = 1
                            dis2pred[2] = 1
                        elif type_map[pre_index] == "SUPPORT":
                            dis2sup[1] = 1
                            dis2sup[2] = 1
                    if data_x[pre_index][1] == "CC":
                        dis2cc[1] = 1
                        dis2cc[2] = 1
                pre_index = i - 3
                if pre_index >= start_index:
                    if pre_index in type_map:
                        if type_map[pre_index] == "PRED":
                            dis2pred[2] = 1
                        elif type_map[pre_index] == "SUPPORT":
                            dis2sup[2] = 1
                    if data_x[pre_index][1] == "CC":
                        dis2cc[2] = 1
                pre_index = i + 1
                if pre_index < data_cnt:
                    if pre_index in type_map:
                        if type_map[pre_index] == "PRED":
                            dis2pred[5] = 1
                            dis2pred[4] = 1
                            dis2pred[3] = 1
                        elif type_map[pre_index] == "SUPPORT":
                            dis2sup[5] = 1
                            dis2sup[4] = 1
                            dis2sup[3] = 1
                    if data_x[pre_index][1] == "CC":
                        dis2cc[5] = 1
                        dis2cc[4] = 1
                        dis2cc[3] = 1
                pre_index = i + 2
                if pre_index < data_cnt:
                    if pre_index in type_map:
                        if type_map[pre_index] == "PRED":
                            dis2pred[4] = 1
                            dis2pred[5] = 1
                        elif type_map[pre_index] == "SUPPORT":
                            dis2sup[4] = 1
                            dis2sup[5] = 1
                    if data_x[pre_index][1] == "CC":
                        dis2cc[4] = 1
                        dis2cc[5] = 1
                pre_index = i + 3
                if pre_index < data_cnt:
                    if pre_index in type_map:
                        if type_map[pre_index] == "PRED":
                            dis2pred[5] = 1
                        elif type_map[pre_index] == "SUPPORT":
                            dis2sup[5] = 1
                    if data_x[pre_index][1] == "CC":
                        dis2cc[5] = 1

                data_x[i].extend(dis2pred)
                data_x[i].extend(dis2sup)
                data_x[i].extend(dis2cc)

                ## Path to pred
                incre = 1
                pred_path = ""
                if pred_index != -1 and pred_index != i:
                    pred_path += "PRED"
                    if pred_index < i:
                        incre = 1
                    else:
                        incre = -1
                    for j in range(pred_index + incre, i, incre):
                        if data_x[j][2][0] == 'B':
                            pred_path += "_" + data_x[j][2][2:]
                
                ## Path to support
                incre = 1
                sup_path = ""
                if sup_index != -1 and sup_index != i:
                    sup_path += "SUPPORT"
                    if sup_index < i:
                        incre = 1
                    else:
                        incre = -1
                    for j in range(sup_index + incre, i, incre):
                        if data_x[j][2][0] == 'B':
                            sup_path += "_" + data_x[j][2][2:]
                
                ## Path to pred
                incre = 1
                cc_path = ""
                if cc_index != -1 and cc_index != i:
                    cc_path += "CC"
                    if cc_index < i:
                        incre = 1
                    else:
                        incre = -1
                    for j in range(cc_index + incre, i, incre):
                        if data_x[j][2][0] == 'B':
                            cc_path += "_" + data_x[j][2][2:]
                data_x[i].append(pred_path)
                data_x[i].append(sup_path)
                data_x[i].append(cc_path)

            start_index = data_cnt
            pred_index = -1
            sup_index = -1
            cc_index = -1
            continue
        fields[0] = ps.stem(fields[0])
        if fields[0].isdigit():
            fields[0] = "NUM"
        ## Add word, POS, BIO
        features = fields[0:3]
        
        data_y.append(0)
        if fields[1] == "CC":
            cc_index = data_cnt
        # features.append(0)
        if len(fields) > 5:
            if fields[5] == "ARG1":
                data_y[data_cnt] = 1
                # features[-1] = 1
            elif fields[5] == "PRED":
                type_map[data_cnt] = "PRED"
                pred_index = data_cnt
            elif fields[5] == "SUPPORT":
                type_map[data_cnt] = "SUPPORT"
                sup_index = data_cnt
        data_x.append(features)
        data_cnt += 1
    # print(data_x[:25])
    # print(data_y)
    return data_x, data_y

def GenerateResult(test_filename, prediction):
    file = open(test_filename, 'r', encoding='UTF-8')
    lines = file.readlines()
    file.close()
    res_file = open('res_2b.txt', 'w', encoding='UTF-8')
    res = []
    index = 0
    max_pred = 0
    max_index = 0
    for line_index in range(len(lines)):
        fields = lines[line_index].split()
        if len(fields) == 0:
            # res[max_index] = res[max_index].strip() + "\tARG1\n"
            # max_index = line_index + 1
            # max_pred = 0
            res.append("\n")
            continue
        res_line = fields[0] + "\t" + fields[1] + "\t" + fields[2] + "\t" + fields[3] + "\t" + fields[4]
        if len(fields) > 5:
            if fields[5] == "SUPPORT":
                res_line = res_line + "\tSUPPORT"
            elif fields[5] == "PRED":
                res_line = res_line + "\tPRED"
        # elif prediction[index] > max_pred:
        #     max_pred = prediction[index]
        #     max_index = line_index
        if prediction[index] > 0.5:
            res_line = res_line + "\tARG1"
        res_line += "\n"
        res.append(res_line)
        index += 1
    res_file.writelines(res)
    res_file.close()

def EncodeData(data):
    le = LabelEncoder()
    for i in range(len(data[0])):
        col = [row[i] for row in data]
        col = le.fit_transform(col).tolist()
        n_class = len(le.classes_)
        # col = [it/n_class for it in col]
        for j in range(len(data)):
            data[j][i] = col[j]
    return np.array(data)

############ Utils End ###################

def run(train_filename, test_filename):

    train_data_x, train_data_y = GenerateDataFromFile_t2b(train_filename)
    test_data_x, test_data_y = GenerateDataFromFile_t2b(test_filename)

    # Batch system
    enc = OneHotEncoder()
    sgd = SGDClassifier()
    print("Data Length: " + str(len(train_data_x)))
    print("Features Length: " + str(len(train_data_x[0])))
    print(train_data_x[:5])

    all_data = train_data_x.copy()
    all_data.extend(test_data_x)
    enc.fit(all_data)
    all_data.clear()

    # batch_data = []
    # batch_y = []
    # classes = np.array([0,1])
    # for i in range(len(train_data_x)):
    #     batch_data.append(train_data_x[i])
    #     batch_y.append(train_data_y[i])
    #     if i % 10000 == 0 or i == len(train_data_x) - 1:
    #         if i == 0:
    #             continue
    #         train_x = enc.transform(batch_data).toarray()
    #         print("Batch: " + str(i/10000) + " Features: " + str(len(train_x[0])))
    #         sgd.partial_fit(train_x, batch_y, classes=classes)
    #         batch_data.clear()
    #         batch_y.clear()

    train_x = enc.transform(train_data_x).toarray()
    sgd.fit(train_x, train_data_y)

    test_x = enc.transform(test_data_x).toarray()
    results = sgd.predict(test_x)
    res_file = open('rawres_2b.txt', 'w', encoding='UTF-8')
    out = []
    for result in results:
        out.append(str(result) + "\n")
    res_file.writelines(out)
    # print(results[:50])
    GenerateResult(test_filename, results)

    
def main(args):
    train_filename = args[1]
    test_filename = args[2]
    run(train_filename, test_filename)

if __name__ == '__main__': sys.exit(main(sys.argv))
