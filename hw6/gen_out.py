import sys

res_filename = 'response.chunk'
test_filename = '%-test'

trn_filename = sys.argv[1]
test_filename = sys.argv[2]

res_file = open(res_filename, 'r', encoding='UTF-8')
test_file = open(test_filename, 'r', encoding='UTF-8')

result_file = open('partitive.txt', 'w', encoding='UTF-8')

res_line = res_file.readline()
test_line = test_file.readline()
out_lines = []

while res_line is not None and res_line != '':
    res_fields = res_line.split()
    test_fields = test_line.split()
    if len(res_fields) == 0:
        out_lines.append("\n")
        res_line = res_file.readline()
        test_line = test_file.readline()
        continue
    out_line = res_fields[0]
    out_line += "\t" + test_fields[1] + "\t" + test_fields[2] + "\t" + test_fields[3] + "\t" + test_fields[4]
    if res_fields[1] != "O":
        out_line += "\t" + res_fields[1]
    out_line += "\n"
    out_lines.append(out_line)
    res_line = res_file.readline()
    test_line = test_file.readline()
    
result_file.writelines(out_lines)

res_file.close()
test_file.close()
result_file.close()