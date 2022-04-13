import sys

trn_filename = '%-training'
test_filename = '%-test'

trn_filename = sys.argv[1]
test_filename = sys.argv[2]

trn_file = open(trn_filename, 'r', encoding='UTF-8')
test_file = open(test_filename, 'r', encoding='UTF-8')

trn_feature = open('training.feature', 'w', encoding='UTF-8')
test_feature = open('test.feature', 'w', encoding='UTF-8')

cnt = 0
predIdx = 0
out_lines = []
sen_lines = []
line = trn_file.readline()
while line is not None and line != '':
    fields = line.split()
    if len(fields) == 0:
        startIdx = len(out_lines) - cnt
        for i in range(cnt):
            out_lines[startIdx + i] = out_lines[startIdx + i].replace('XXX', str(abs(i-predIdx)))
        out_lines.append('\n')
        sen_lines = []
        cnt = 0
        predIdx = 0
        line = trn_file.readline()
        continue
    out_line = fields[0]
    out_line += "\tPOS=" + fields[1]
    out_line += "\tBIO=" + fields[2]
    if len(sen_lines) > 0:
        out_line += "\tPrevious_POS=" + sen_lines[cnt - 1][1]
        out_line += "\tPrevious_BIO=" + sen_lines[cnt - 1][2]
    out_line += "\tDisToPred=XXX"
    if len(fields) > 5:
        out_line += "\t" + fields[5]
        if fields[5] == "PRED":
            predIdx = cnt
    else:
        out_line += "\tO"
    out_line += "\n"
    out_lines.append(out_line)
    sen_lines.append(fields)
    cnt += 1
    line = trn_file.readline()
    
trn_feature.writelines(out_lines)

cnt = 0
predIdx = 0
out_lines = []
sen_lines = []
line = test_file.readline()
while line is not None and line != '':
    fields = line.split()
    if len(fields) == 0:
        startIdx = len(out_lines) - cnt
        for i in range(cnt):
            out_lines[startIdx + i] = out_lines[startIdx + i].replace('XXX', str(abs(i-predIdx)))
        out_lines.append('\n')
        sen_lines = []
        cnt = 0
        predIdx = 0
        line = test_file.readline()
        continue
    out_line = fields[0]
    out_line += "\tPOS=" + fields[1]
    out_line += "\tBIO=" + fields[2]
    if len(sen_lines) > 0:
        out_line += "\tPrevious_POS=" + sen_lines[cnt - 1][1]
        out_line += "\tPrevious_BIO=" + sen_lines[cnt - 1][2]
    out_line += "\tDisToPred=XXX"
    if len(fields) > 5:
        if fields[5] == "PRED":
            predIdx = cnt
    out_line += "\n"
    out_lines.append(out_line)
    sen_lines.append(fields)
    cnt += 1
    line = test_file.readline()
    
test_feature.writelines(out_lines)


trn_file.close()
test_file.close()
trn_feature.close()
test_feature.close()