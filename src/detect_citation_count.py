# -*- encoding:utf-8 -*-
if __name__ == '__main__':
    with open('ans_new_tra.csv','wb+') as w:
        with open('./features/tra_236.csv', 'rb') as rf2, open('./features/train.csv', 'rb') as rf1:
            lines1 = rf1.readlines()
            lines2 = rf2.readlines()
            lines1 = lines1[1:]
            lines2 = lines2[1:]
            cnt_citation = 0
            cnt = 0
            cnt_label = 0
            cnt_e = 0
            ans = list()
            i = 1
	    info = list()
            for inx1, inx2 in zip(lines1, lines2):
                inx1 = inx1.decode(encoding='utf-8').strip().split(',')
                inx2 = inx2.decode(encoding='utf-8').strip().split(',')
                # print inx1[1]
                # print inx2[29]
                if int(inx2[3]) == 0:
                    cnt_citation += 1
                if int(inx1[1]) == 0:
                    cnt_label += 1
                if int(inx2[3]) == 0 and int(inx1[1]) == 0:
                    cnt += 1
                    ans.append(i)
                if int(inx2[3]) != 0 and int(inx1[1]) == 0:
                    cnt_e += 1
                    info = inx1[0] + ',' + inx1[1] + ',' + inx2[3]
                    #info = ','.join(info)
		    w.write(info.encode(encoding='utf-8'))
                    w.write('\r\n'.encode(encoding='utf-8'))
            print '引用次数为0',
            print cnt_citation
            print '标签为0',
            print cnt_label
            print '同时为0',
            print cnt
            print cnt_e

                                                                                                             
