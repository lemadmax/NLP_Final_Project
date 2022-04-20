from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
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
                data_x[data_index].append(abs(data_index - sup_index))
                data_x[data_index].append(abs(data_index - pred_index))
            sup_index = -1
            pred_index = -1
            continue
        if fields[3] == "0":
            start_index = data_cnt
        ## Add POS, BIO, NUM features
        features = fields[1:3]

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

def GenerateResult(test_filename, prediction):
    file = open(test_filename, 'r', encoding='UTF-8')
    lines = file.readlines()
    file.close()
    res_file = open('res.txt', 'w', encoding='UTF-8')
    res = []
    index = 0
    for line in lines:
        fields = line.split()
        if len(fields) == 0:
            res.append("\n")
            continue
        res_line = fields[0] + "\t" + fields[1] + "\t" + fields[2] + "\t" + fields[3] + "\t" + fields[4]
        if prediction[index] != "O":
            res_line = res_line + "\t" + prediction[index]
        elif len(fields) > 5:
            if fields[5] == "SUPPORT":
                res_line = res_line + "\tSUPPORT"
            elif fields[5] == "PRED":
                res_line = res_line + "\tPRED"
        res_line += "\n"
        res.append(res_line)
        index += 1
    res_file.writelines(res)
    res_file.close()

############ Utils End ###################

def run(train_filename, test_filename):
    
    train_data_x, train_data_y = GenerateDataFromFile(train_filename)
    test_data_x, test_data_y = GenerateDataFromFile(test_filename)

    print(train_data_x[0])
    enc = OneHotEncoder()
    train_X = enc.fit_transform(train_data_x).toarray()
    test_X = enc.fit_transform(test_data_x).toarray()
    print(len(train_X[0]))

    model = LinearRegression()
    model.fit(train_X, train_data_y)
    
    results = model.predict(test_X[:,:])
    prediction = []
    for result in results:
        if result > 1.5:
            prediction.append("ARG1")
        else:
            prediction.append("O")
    # print(prediction)
    # print(data_y[:50])
    # print(data_y[:4])
    GenerateResult(test_filename, prediction)

    
def main(args):
    train_filename = args[1]
    test_filename = args[2]
    run(train_filename, test_filename)

if __name__ == '__main__': sys.exit(main(sys.argv))
