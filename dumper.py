'''
Xor Dataset Generator : Python 3
Author: Abhishek Munagekar
'''
import random
import time
#Parameters

xor_num = 5
nos_entry = 32

dumpfilename='dump.txt'

string = ['0','1']
random.seed()
dump = open(dumpfilename,'w')
dump.write(str(nos_entry)+"\n")
for i in range (nos_entry):
	_input = [None] * xor_num
	_output = 0
	for j in range(xor_num):
		if i & (1 << j):
			_input[j] = 1
		else:
			_input[j] = 0
		# _input[j]=random.randint(0,1)
		dump.write(string[_input[j]]+"\t")
		_output ^=_input[j]
	if _output == 0:
		dump.write(string[1]+"\t")
		dump.write(string[0]+"\n")
	else:
		dump.write(string[0]+"\t")
		dump.write(string[1]+"\n")
dump.close()
