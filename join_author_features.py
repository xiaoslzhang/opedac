import sys

if len(sys.argv) != 4:
    print("python join_author_features.py inputfile1 inputfile2 outputfile ")
    exit()
    
if __name__ == '__main__':
        with open(sys.argv[3], 'wb+') as wf:
            with open(sys.argv[1], 'rb') as rf1, open(sys.argv[2]) as rf2:
                lines1 = rf1.readlines()
                lines2 = rf2.readlines()
                for inx1, inx2 in zip(lines1, lines2):
                    inf = list()
                    inx1 = inx1.decode(encoding = 'utf-8').strip().split(',')
                    inx2 = inx2.decode(encoding = 'utf-8').strip().split(',')
                    inf = inx1 + inx2[1:]
                    inf = ','.join(inf)
                    wf.write(inf.encode(encoding = 'utf-8'))
                    wf.write('\r\n'.encode(encoding = 'utf-8'))                                                                                                            
