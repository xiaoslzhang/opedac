# -*- coding: utf-8 -*-
# 对task3结果进行处理，得到sample_sumission.txt
import os
import sys

if len(sys.argv) != 2:
    print("python get_result_txt.py inputfile ")
    exit()

with open(sys.argv[1],'r') as file_tk3, open('sample_submission.txt','r') as file_ans:
    content_add = file_tk3.read()
    content_add = content_add.decode(encoding='utf-8')
    content = file_ans.read()
    content = content.decode(encoding='utf-8')
    pos = content.find('authorname\tcitation')
    file_tk3.close()
    if pos != -1:
        content = content[:pos] + content_add
        with open('sample_submission.txt', 'w') as file_w:
            file_w.write(content.encode(encoding='utf-8'))
            file_w.write('</task3>'.encode(encoding='utf-8'))
    print'success'
    
