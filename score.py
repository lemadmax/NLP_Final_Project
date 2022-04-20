## COMMAND: python score.py key.txt response.txt

import sys

# key_out = open('Key_out.txt', 'w', encoding='UTF-8')
# res_out = open('res_out.txt', 'w', encoding='UTF-8')

def readfile(keyFileName, responseFileName):
	keyFile = open(keyFileName, 'r')
	key = keyFile.readlines()
	responseFile = open(responseFileName, 'r')
	response = responseFile.readlines()

	n_key = []
	for line in key:
		fields = line.split()
		if len(fields) == 0:
			n_key.append('\n')
			continue
		n_line = fields[0]
		if len(fields) > 5:
			n_line += "\t" + fields[5]
		else:
			n_line += "\tO"
		n_line += "\n"
		n_key.append(n_line)
		
	n_response = []
	for line in response:
		fields = line.split()
		if len(fields) == 0:
			n_response.append('\n')
			continue
		n_line = fields[0]
		if len(fields) > 5:
			n_line += "\t" + fields[5]
		else:
			n_line += "\tO"
		n_line += "\n"
		n_response.append(n_line)

	# key_out.writelines(n_key)
	# res_out.writelines(n_response)
	# key_out.close()
	# res_out.close()
	
	return n_key, n_response

def score (keyFileName, responseFileName):
	key, response = readfile(keyFileName, responseFileName)
	if len(key) != len(response):
		print("length mismatch between key and submitted file")
		exit()
	correct = 0
	incorrect = 0
	keyGroupCount = 0
	responseGroupCount = 0
	correctGroupCount = 0
	for i in range(len(key)):
		key[i] = key[i].rstrip('\n')
		response[i] = response[i].rstrip('\n')
		if key[i] == "":
			if response[i] == "":
				continue
			else:
				print("sentence break expected at line " + str(i))
				exit()
		keyFields = key[i].split('\t')
		if len(keyFields) != 2:
			print("format error in key at line " + str(i) + ":" + key[i])
			exit()
		keyToken = keyFields[0]
		keyTag = keyFields[1]
		responseFields = response[i].split('\t')
		if len(responseFields) != 2:
			print("format error at line " + str(i))
			exit()
		responseToken = responseFields[0]
		responseTag = responseFields[1]
		if responseToken != keyToken:
			print("token mismatch at line " + str(i))
			print("ResponseToken: " + str(responseToken))
			print("KeyToken: " + str(keyToken))
			exit()
		if responseTag == keyTag:
			correct = correct + 1
		else:
			incorrect = incorrect + 1
		if responseTag == 'ARG1':
			responseGroupCount = responseGroupCount + 1
		if keyTag == 'ARG1':
			keyGroupCount = keyGroupCount + 1
		if responseTag == 'ARG1' and responseTag == keyTag:
			correctGroupCount = correctGroupCount + 1
	print(correct, "out of", str(correct + incorrect) + " tags correct")
	accuracy = 100.0 * correct / (correct + incorrect)
	print("  accuracy: %5.2f" % accuracy)
	print(keyGroupCount, "groups in key")
	print(responseGroupCount, "groups in response")
	print(correctGroupCount, "correct groups")
	precision = 100.0 * correctGroupCount / responseGroupCount
	recall = 100.0 * correctGroupCount / keyGroupCount
	F = 2 * precision  * recall / (precision + recall)
	print("  precision: %5.2f" % precision)
	print("  recall:    %5.2f" % recall)
	print("  F1:        %5.2f" % F)

def main(args):
	key_file = args[1]
	response_file = args[2]
	score(key_file,response_file)

if __name__ == '__main__': sys.exit(main(sys.argv))