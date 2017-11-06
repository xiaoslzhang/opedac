# -*- coding: utf-8 -*-
import sys
if len(sys.argv) != 3:
    print("用法:python3 get_features.py train.csv/validation.csv output.csv")
    exit()


from functools import reduce
import numpy as np
from load_2014 import papers, authors, confs
from pip.utils.ui import DownloadProgressBar
print("加载信息完成，开始提取特征 ...")
print("正在处理期刊数据 ...")

#CLAZZ_NUM = 3  # 期刊根据评价分数进行分类的类数
FEATURES_NUM = 12  # 特征总维数

def get_features(author_name, head=False):
    feature_head = "author_name,citation_count,paper_count,ref_count,recent_num_coauthors,mean_citations_per_year,mean_papers_per_year,mean_refs_per_year,total_num_venues"
    feature_head += ",first_author_paper_count,first_author_citation_count"
    feature_head += ",coauthor_paper_count,coauthor_citation_count"
    
    if head:
        return feature_head + "\n"
    features = {
        'author_name': author_name,
        'citation_count': 0,
        'paper_count': 0,
        'ref_count': 0,
        'recent_num_coauthors': 0,
        'mean_citations_per_year': 0,
        'mean_papers_per_year': 0,
        'mean_refs_per_year': 0,
        'total_num_venues': 0,
        'first_author_paper_count': 0,
        'first_author_citation_count': 0,
        'coauthor_paper_count': 0,
        'coauthor_citation_count': 0
    }

    # 计算全部作者全部论文的指标
    if author_name not in authors:
        return author_name + "," + ",".join(['0'] * FEATURES_NUM) + "\n"
    author_papers = list(
        map(lambda p_index: papers[p_index], authors[author_name]['papers']))
    all_citation = list(map(lambda paper: len(paper["be_ref"]), author_papers))
    all_ref = list(map(lambda paper: len(paper["ref"]), author_papers))
    features['citation_count'] = sum(all_citation)
    features['paper_count'] = len(author_papers)
    features['ref_count'] = sum(all_ref)

    author_papers_last_two_years = list(
        filter(lambda paper: paper["year"] > 2011, author_papers))
    features['recent_num_coauthors'] = sum(
        [(len(paper['author']) - 1) for paper in author_papers_last_two_years])

    age = 2016 - 2014 + 1
    features['mean_citations_per_year'] = features['citation_count'] / age
    features['mean_refs_per_year'] = features['ref_count'] / age
    features['mean_papers_per_year'] = features['paper_count'] / age
    features['total_num_venues'] = len(authors[author_name]["confs"])

    first_author_papers = list(
        filter(lambda paper: paper["author"].index(author_name) == 0, author_papers))
    features["first_author_paper_count"] = len(first_author_papers)
    features["first_author_citation_count"] = sum(
        list(map(lambda paper: len(paper["be_ref"]), first_author_papers)))

    author_papers_in_five_year = list(
        filter(lambda paper: paper["year"] > 2013, author_papers))
    coauthor_in_five_year = list(
        map(lambda paper: paper["author"], author_papers_in_five_year))
    if len(coauthor_in_five_year) != 0:
        coauthor_in_five_year = list(
            reduce(lambda x, y: x + y, coauthor_in_five_year))
        coauthor_in_five_year.remove(author_name)
    coauthor_papers_in_five_year = [authors[_author]["papers"]
                                    for _author in coauthor_in_five_year if _author in authors]
    if len(coauthor_papers_in_five_year) != 0:
        coauthor_papers_in_five_year = map(lambda p_index: papers[p_index], list(
            set(reduce(lambda x, y: x + y, coauthor_papers_in_five_year))))
    features["coauthor_paper_count"] = len(coauthor_in_five_year)
    features["coauthor_citation_count"] = sum(
        list(map(lambda paper: len(paper["be_ref"]), coauthor_papers_in_five_year)))

    return ",".join([str(features[feature]) for feature in feature_head.split(",")]) + "\n"

dataset = open(sys.argv[1]).read().split("\n")[1:]
output = open(sys.argv[2], "a")
all_count = len(dataset)
bar = DownloadProgressBar(max=all_count - 1)
value = 0
output.write(get_features("", head=True))
print("正在生成全部特征 ...")
for i in bar.iter(range(all_count - 1)):
    line = dataset[value]
    author_name = line.split(",")[0]
    if author_name != "":
        feature = get_features(author_name)
    output.write(feature)
    value += 1


def check_dimension():
    s = open(sys.argv[2], "r")
    print("正在检查维数")
    while True:
        line = s.readline()
        if line == "":
            break
        else:
            if len(line.strip().split(",")) != FEATURES_NUM + 1:
                print("error:input-" + str(FEATURES_NUM + 1) +
                      ",actual-" + str(len(line.strip().split(","))))
                print(line)
                break
    else:
        print("维数检查完毕")


check_dimension()
