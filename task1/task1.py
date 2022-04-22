from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
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
                data_y[data_cnt] = 2
        data_cnt += 1
    # print(data_x[:25])
    # print(data_y)
    return data_x, data_y

def GenerateDataFromFile_t1(filename):
    file = open(filename, 'r', encoding='UTF-8')
    lines = file.readlines()
    file.close()
    # sample_cnt = len(lines)
    # print(sample_cnt)
    data_x = []
    data_y = []

    start_index = 0
    data_cnt = 0
    for index in range(len(lines)):
        fields = lines[index].split()
        if len(fields) == 0:
            
            for i in range(start_index, data_cnt):
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
                if data_x[i][1][0] == 'N' and data_x[i+1][2][0] != 'I':
                    data_x[i].append(1)
                else:
                    data_x[i].append(0)

            start_index = data_cnt
            continue
        ## Add word, POS, BIO
        features = fields[0:3]
        
        data_x.append(features)
        data_y.append(0)
        if len(fields) > 5 and fields[5] == "ARG1":
            data_y[data_cnt] = 1
        data_cnt += 1
    # print(data_x[:25])
    # print(data_y)
    return data_x, data_y

def GenerateResult(test_filename, prediction):
    file = open(test_filename, 'r', encoding='UTF-8')
    lines = file.readlines()
    file.close()
    res_file = open('res.txt', 'w', encoding='UTF-8')
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
    
    # train_data_x, train_data_y = GenerateDataFromFile(train_filename)
    # test_data_x, test_data_y = GenerateDataFromFile(test_filename)

    # enc = OneHotEncoder()
    # train_X = enc.fit_transform(train_data_x).toarray()
    # print(len(train_X[0]))
    # print(train_X[0])

    # print(train_X)

    # model = LinearRegression()
    # model.fit(train_X, train_data_y)

    # test_X = enc.fit_transform(test_data_x).toarray()
    # results = model.predict(test_X[:,:])
    # print(results[:50])
    # GenerateResult(test_filename, results)
    train_data_x, train_data_y = GenerateDataFromFile_t1(train_filename)
    test_data_x, test_data_y = GenerateDataFromFile_t1(train_filename)

    # train_x = EncodeData(train_data_x)
    # test_x = EncodeData(test_data_x)
    enc = OneHotEncoder()
    train_x = enc.fit_transform(train_data_x).toarray()
    test_x = enc.fit_transform(test_data_x).toarray()
    print(len(train_x[0]))
    # for i in range(25):
    #     print(train_x[i])

    model = LinearRegression()
    model.fit(train_x, train_data_y)
    results = model.predict(test_x[:,:])
    print(results[:50])
    GenerateResult(test_filename, results)

    
def main(args):
    train_filename = args[1]
    test_filename = args[2]
    run(train_filename, test_filename)

if __name__ == '__main__': sys.exit(main(sys.argv))
