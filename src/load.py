from pip.utils.ui import DownloadProgressBar
from math import ceil
print("正在载入 papers.txt ...")
papers = open("./papers.txt").read().split("\n\n")
print("载入完成，正在构建 hash")


def handlePaperInfo(s):
    paper = s.split("\n")
    index = None
    rs = {'author': None, 'year': None, 'conf': None, 'ref': [], 'be_ref': []}
    for info in paper:
        if info[:2] == '#i':
            index = int(info[6:])
        elif info[:2] == '#@':
            rs['author'] = info[2:].split(",")
        elif info[:2] == '#t':
            rs['year'] = int(info[2:])
        elif info[:2] == '#c':
            rs['conf'] = info[2:]
        elif info[:2] == '#%':
            rs['ref'].append(int(info[2:]))
    if rs['year'] == None:print(s)
    if rs['year'] > 2013: return False
    return index, rs

paper_dict = dict()
all_count = len(papers)
bar_count = int(ceil(len(papers)/1000))
bar = DownloadProgressBar(max=bar_count)
_i = 0
for i in bar.iter(range(bar_count)):
    for j in range(_i*1000,(_i+1)*1000):
        if j == all_count:
            break
        paper = handlePaperInfo(papers[j])
        if paper == False:
            continue
        paper_dict[paper[0]] = paper[1]
    _i += 1
papers = paper_dict

print("正在构建被引用关系")
cnt = 0
for p_index in papers.keys():
    paper = papers[p_index]
    if "ref" in paper:
        for r_index in paper["ref"]:
            if r_index not in papers:
                papers[p_index]["ref"] = list(filter(lambda i:r_index != i,papers[p_index]["ref"]))
           # elif papers[r_index]["year"] > papers[p_index]["year"]:
           #     cnt += 1
           #     papers[p_index]["ref"] = list(filter(lambda i:r_index != i,papers[p_index]["ref"]))
            else:
                papers[r_index]["be_ref"].append(p_index)

print("移除了{}个不正确的引用关系".format(cnt))

print("构建 hash 完成，正在生成作者-论文关系表及期刊-论文表")


authors = {}
confs = {}

for p_index in papers.keys():
    for author in papers[p_index]['author']:
        if author not in authors:
            authors[author] = {'papers':[],'confs':set()}
        authors[author]['papers'].append(p_index)
        authors[author]['confs'].add(papers[p_index]['conf'])
    if papers[p_index]['conf'] not in confs:
        confs[papers[p_index]['conf']] = {'papers':[]}
    confs[papers[p_index]['conf']]['papers'].append(p_index)

print("作者-论文关系表及期刊-论文表构建完毕")
