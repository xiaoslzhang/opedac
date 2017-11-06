import sys
if len(sys.argv) != 3:
    print("用法:python3 get_features.py train.csv/validation.csv output.csv")
    exit()


from functools import reduce
import numpy as np
from load_2016 import papers, authors, confs
from pip.utils.ui import DownloadProgressBar
print("加载信息完成，开始提取特征 ...")
print("正在处理期刊数据 ...")

CLAZZ_NUM = 40  # 期刊根据评价分数进行分类的类数
FEATURES_NUM = 224  # 特征总维数


def get_conf_info(conf_name):
    # 计算期刊全部年份的特征
    conf_papers = list(
        map(lambda p_index: papers[p_index], confs[conf_name]['papers']))
    all_citation = list(map(lambda paper: len(paper['be_ref']), conf_papers))
    confs[conf_name]["citation_count"] = sum(all_citation)
    h_index = 0
    all_citation.sort(reverse=True)
    confs[conf_name]["max_single_paper_citations"] = all_citation[0]
    for i in all_citation:
        if h_index < i:
            h_index += 1
        else:
            break
    confs[conf_name]["h_index"] = h_index
    confs[conf_name]["papers_cnt"] = len(confs[conf_name]["papers"])
    if confs[conf_name]["papers_cnt"] != 0:
        confs[conf_name]["citation_per_paper"] = confs[conf_name]["citation_count"] / \
            confs[conf_name]["papers_cnt"]
    else:
        confs[conf_name]["citation_per_paper"] = 0

    # 计算两年前的期刊特征
    conf_papers_two_years_ago = list(
        filter(lambda paper: paper["year"] <= 2014, conf_papers))
    citation_two_years_ago = list(
        map(lambda paper: len(paper['be_ref']), conf_papers_two_years_ago))
    h_index_two_years_ago = 0
    citation_two_years_ago.sort(reverse=True)
    for i in citation_two_years_ago:
        if h_index_two_years_ago < i:
            h_index_two_years_ago += 1
        else:
            break
    confs[conf_name]["h_index_delta"] = h_index - h_index_two_years_ago
    if len(conf_papers_two_years_ago) != 0:
        confs[conf_name]["mean_citations_per_paper_delta"] = confs[conf_name]["citation_per_paper"] - \
            (sum(citation_two_years_ago) / len(conf_papers_two_years_ago))
    else:
        confs[conf_name]["mean_citations_per_paper_delta"] = 0
    confs[conf_name]["papers_delta"] = len(
        conf_papers) - len(conf_papers_two_years_ago)


def get_features(author_name, head=False):
    feature_head = "author_name,h_index,h_index_delta,citation_count,citation_delta,mean_citations_per_paper,mean_citations_per_paper_delta,mean_venue_h_index,min_venue_h_index,max_venue_h_index,mean_venue_h_index_delta,min_venue_h_index_delta,max_venue_h_index_delta,mean_venue_citation_count,min_venue_citation_count,max_venue_citation_count,mean_venue_citation_delta,min_venue_citation_delta,max_venue_citation_delta,career_age,papers_cnt,mean_papers_per_year,mean_citations_per_year,recent_num_coauthors,papers_delta,max_single_paper_citations,venue_papers_mean,venue_papers_min,venue_papers_max,venue_papers_delta_mean,venue_papers_delta_min,venue_papers_delta_max,total_num_venues,venue_rank_mean,venue_rank_min,venue_rank_max,venue_max_single_paper_citations_mean,venue_max_single_paper_citations_min,venue_max_single_paper_citations_max"
    feature_head += ',' + \
        ','.join(list(map(lambda x: str(x) + '_cnt', range(1997, 2017))))
    feature_head += ',' + \
        ','.join(list(map(lambda x: str(x) + '_citation', range(1997, 2017))))
    feature_head += ',' + \
        ','.join(list(map(lambda x: str(x) + '_conf', range(CLAZZ_NUM))))
    # 作为第一作者发论文的总数,作为第一作者发论文的被引用数
    feature_head += ',first_author_paper_cnt,first_author_citation'
    # 作为第一作者在各个年份发文数与被引数
    feature_head += ',' + \
        ','.join(list(map(lambda x: 'first_author_' +
                          str(x) + '_cnt', range(1997, 2017))))
    feature_head += ',' + \
        ','.join(list(map(lambda x:  'first_author_' +
                          str(x) + '_citation', range(1997, 2017))))
    # 作为第一作者在各个等级期刊发文数与被引数
    feature_head += ',' + \
        ','.join(list(map(lambda x: 'first_author_' +
                          str(x) + '_conf', range(CLAZZ_NUM))))

    # 近 5 年以第一作者发表的论文数，近五年来以第一作者发表的论文引用次数，近两年来以第一作者发表论文被引用次数之差，近两年来以第一作者发表论文数之差
    feature_head += ",five_year_first_author_paper_count,five_year_first_author_citation_count,first_author_paper_count_delta,first_author_citation_count_delta"

    # 近 5 年来合作作者论文数，近 5 年来合作作者论文的引用次数，近两年来合作作者论文数之差，近两年来合作作者论文引用次数之差
    feature_head += ",coauthor_paper_count,coauthor_citation_count,coauthor_paper_count_delta,coauthor_citation_count_delta"

    # 作者排名，作者两年内排名发生的变化，以第一作者发表的论文的作者排名，近两年来以第一作者发表论文的排名之差
    feature_head += ",author_rank,author_rank_delta,first_author_rank,first_author_rank_delta"

    # 合作作者排名均值、最小值、最大值
    feature_head += ",mean_coauthor_rank,min_coauthor_rank,max_coauthor_rank"

    # 近两年来合作作者排名之差的均值、最小值、最大值
    feature_head += ",mean_coauthor_rank_delta,min_coauthor_rank_delta,max_coauthor_rank_delta"

    # 把前面的 citation 换成 ref 再来一次
    feature_head += ",ref_count,ref_delta,mean_refs_per_paper,mean_refs_per_paper_delta,mean_refs_per_year,max_single_paper_refs"

    if head:
        return feature_head + "\n"
    features = {
        'author_name': author_name,
        'h_index': 0,
        'h_index_delta': 0,
        'citation_count': 0,
        'citation_delta': 0,
        'mean_citations_per_paper': 0,
        'mean_citations_per_paper_delta': 0,
        'mean_venue_h_index': 0,
        'min_venue_h_index': 0,
        'max_venue_h_index': 0,
        'mean_venue_h_index_delta': 0,
        'min_venue_h_index_delta': 0,
        'max_venue_h_index_delta': 0,
        'mean_venue_citation_count': 0,
        'min_venue_citation_count': 0,
        'max_venue_citation_count': 0,
        'mean_venue_citation_delta': 0,
        'min_venue_citation_delta': 0,
        'max_venue_citation_delta': 0,
        'career_age': 0,
        'papers_cnt': 0,
        'mean_papers_per_year': 0,
        'mean_citations_per_year': 0,
        'recent_num_coauthors': 0,
        'papers_delta': 0,
        'max_single_paper_citations': 0,
        'venue_papers_mean': 0,
        'venue_papers_min': 0,
        'venue_papers_max': 0,
        'venue_papers_delta_mean': 0,
        'venue_papers_delta_min': 0,
        'venue_papers_delta_max': 0,
        'total_num_venues': 0,
        'venue_rank_mean': 0,
        'venue_rank_min': 0,
        'venue_rank_max': 0,
        'venue_max_single_paper_citations_mean': 0,
        'venue_max_single_paper_citations_min': 0,
        'venue_max_single_paper_citations_max': 0
    }

    # 计算全部作者全部论文的指标
    if author_name not in authors:
        return author_name + "," + ",".join(['0'] * FEATURES_NUM) + "\n"
    author_papers = list(
        map(lambda p_index: papers[p_index], authors[author_name]['papers']))
    all_citation = list(map(lambda paper: len(paper["be_ref"]), author_papers))
    all_ref = list(map(lambda paper: len(paper["ref"]), author_papers))
    features['papers_cnt'] = len(author_papers)
    features['citation_count'] = sum(all_citation)
    features['ref_count'] = sum(all_ref)
    if len(author_papers) != 0:
        features['mean_citations_per_paper'] = features['citation_count'] / \
            len(author_papers)
        features['mean_refs_per_paper'] = features['ref_count'] / \
            len(author_papers)
    else:
        features['mean_citations_per_paper'] = 0
        features['mean_refs_per_paper'] = 0
    h_index = 0
    all_citation.sort(reverse=True)
    all_ref.sort(reverse=True)
    features['max_single_paper_citations'] = all_citation[0]
    features['max_single_paper_refs'] = all_ref[0]
    for i in all_citation:
        if h_index < i:
            h_index += 1
        else:
            break
    features['h_index'] = h_index
    author_papers.sort(key=lambda paper: paper["year"])
    features['career_age'] = 2016 - author_papers[0]["year"] + 1
    features['mean_citations_per_year'] = features['citation_count'] / \
        features['career_age']
    features['mean_refs_per_year'] = features['ref_count'] / \
        features['career_age']
    features['mean_papers_per_year'] = features['papers_cnt'] / \
        features['career_age']
    features['total_num_venues'] = len(authors[author_name]["confs"])
    features["author_rank"] = features['citation_count'] / \
        features['papers_cnt'] / features['career_age']

    # 五年内的一些特征
    author_papers_in_five_year = list(
        filter(lambda paper: paper["year"] > 2011, author_papers))
    first_author_papers_in_five_year = list(filter(
        lambda paper: paper["author"].index(author_name) == 0, author_papers_in_five_year))
    features["five_year_first_author_paper_count"] = len(
        first_author_papers_in_five_year)
    features["five_year_first_author_citation_count"] = sum(
        list(map(lambda paper: len(paper["be_ref"]), first_author_papers_in_five_year)))
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

    # 两年内的论文的各种指标
    author_papers_last_two_years = list(
        filter(lambda paper: paper["year"] > 2014, author_papers))
    citation_last_two_years = list(
        map(lambda x: len(x["be_ref"]), author_papers_last_two_years))
    ref_last_two_years = list(
        map(lambda x: len(x["ref"]), author_papers_last_two_years))
    features['papers_delta'] = len(author_papers_last_two_years)
    features['recent_num_coauthors'] = sum(
        [(len(paper['author']) - 1) for paper in author_papers_last_two_years])
    features['citation_delta'] = sum(citation_last_two_years)
    features['ref_delta'] = sum(ref_last_two_years)
    first_author_papers_in_two_year = list(filter(
        lambda paper: paper["author"].index(author_name) == 0, author_papers_last_two_years))
    features["first_author_paper_count_delta"] = len(
        first_author_papers_in_two_year)
    features["first_author_citation_count_delta"] = sum(
        list(map(lambda paper: len(paper["be_ref"]), first_author_papers_in_two_year)))

    coauthor_in_two_year = list(map(lambda paper: paper["author"], author_papers_last_two_years))
    if len(coauthor_in_two_year) != 0:
        coauthor_in_two_year = list(
            reduce(lambda x, y: x + y, coauthor_in_two_year))
        coauthor_in_two_year.remove(author_name)

    coauthor_papers_in_two_year = [authors[_author]["papers"]
                                   for _author in coauthor_in_two_year if _author in authors]
    if len(coauthor_papers_in_two_year) != 0:
        coauthor_papers_in_two_year = list(
            set(reduce(lambda x, y: x + y, coauthor_papers_in_two_year)))
    if len(coauthor_in_two_year) != 0:
        coauthor_papers_in_two_year = list(map(
            lambda p_index: papers[p_index], coauthor_papers_in_two_year))

    features["coauthor_paper_count_delta"] = len(
        coauthor_papers_in_two_year) - features["coauthor_paper_count"]
    features["coauthor_citation_count_delta"] = sum(
        list(map(lambda paper: len(paper["be_ref"]), coauthor_papers_in_two_year))) - features["coauthor_citation_count"]

    # 作为第一作者发论文的总数与引用数
    first_author_papers = list(
        filter(lambda paper: paper["author"].index(author_name) == 0, author_papers))
    features["first_author_paper_cnt"] = len(first_author_papers)
    features["first_author_citation"] = sum(
        list(map(lambda paper: len(paper["be_ref"]), first_author_papers)))
    if features['first_author_paper_cnt'] != 0:
        features["first_author_rank"] = features['first_author_citation'] / \
            features['first_author_paper_cnt'] / features['career_age']
    else:
        features["first_author_rank"] = 0

    # 两年前的论文指标
    author_papers_two_years_ago = list(
        filter(lambda paper: paper["year"] <= 2014, author_papers))
    citation_two_years_ago = list(
        map(lambda x: len(x["be_ref"]), author_papers_two_years_ago))
    ref_two_years_ago = list(
        map(lambda x: len(x["ref"]), author_papers_two_years_ago))
    h_index_two_years_ago = 0
    citation_two_years_ago.sort(reverse=True)
    for i in citation_two_years_ago:
        if h_index_two_years_ago < i:
            h_index_two_years_ago += 1
        else:
            break
    features['h_index_delta'] = features['h_index'] - h_index_two_years_ago
    if len(author_papers_two_years_ago) != 0:
        features['mean_citations_per_paper_delta'] = features['mean_citations_per_paper'] - \
            (sum(citation_two_years_ago) / len(author_papers_two_years_ago))
        features['mean_refs_per_paper_delta'] = features['mean_refs_per_paper'] - \
            (sum(ref_two_years_ago) / len(author_papers_two_years_ago))
        if len(author_papers_two_years_ago) != 0:
            features["author_rank_delta"] = features["author_rank"] - \
                (sum(citation_two_years_ago) /
                 len(author_papers_two_years_ago) / (features['career_age'] - 2))
        else:
            features["author_rank_delta"] = 0
        first_author_papers_two_years_ago = list(filter(
            lambda paper: paper["author"].index(author_name) == 0, author_papers_two_years_ago))
        citation_first_author_papers_two_years_ago = sum(
            list(map(lambda paper: len(paper["be_ref"]), first_author_papers_two_years_ago)))

        if len(first_author_papers_two_years_ago) != 0:
            features["first_author_rank_delta"] = features["first_author_rank"] - \
                (citation_first_author_papers_two_years_ago /
                 len(first_author_papers_two_years_ago) / (features['career_age'] - 2))
        else:
            features["first_author_rank_delta"] = features["first_author_rank"]
    else:
        features['mean_citations_per_paper_delta'] = 0
        features['mean_refs_per_paper_delta'] = 0
        features["author_rank_delta"] = features["author_rank"]
        features["first_author_rank_delta"] = features["first_author_rank"]

    # 期刊特征
    for conf_name in authors[author_name]['confs']:
        if 'h_index' not in confs[conf_name]:
            get_conf_info(conf_name)

    author_confs = list(
        map(lambda conf_name: confs[conf_name], authors[author_name]['confs']))
    features['venue_papers_mean'] = np.mean(
        list(map(lambda conf: conf["papers_cnt"], author_confs)))
    features['venue_papers_max'] = np.max(
        list(map(lambda conf: conf["papers_cnt"], author_confs)))
    features['venue_papers_min'] = np.min(
        list(map(lambda conf: conf["papers_cnt"], author_confs)))

    features['venue_papers_delta_mean'] = np.mean(
        list(map(lambda conf: conf["papers_delta"], author_confs)))
    features['venue_papers_delta_max'] = np.max(
        list(map(lambda conf: conf["papers_delta"], author_confs)))
    features['venue_papers_delta_min'] = np.min(
        list(map(lambda conf: conf["papers_delta"], author_confs)))

    features['venue_rank_mean'] = np.mean(
        list(map(lambda conf: conf["citation_per_paper"], author_confs)))
    features['venue_rank_max'] = np.max(
        list(map(lambda conf: conf["citation_per_paper"], author_confs)))
    features['venue_rank_min'] = np.min(
        list(map(lambda conf: conf["citation_per_paper"], author_confs)))

    features['venue_max_single_paper_citations_mean'] = np.mean(
        list(map(lambda conf: conf["max_single_paper_citations"], author_confs)))
    features['venue_max_single_paper_citations_max'] = np.max(
        list(map(lambda conf: conf["max_single_paper_citations"], author_confs)))
    features['venue_max_single_paper_citations_min'] = np.min(
        list(map(lambda conf: conf["max_single_paper_citations"], author_confs)))

    features['mean_venue_h_index'] = np.mean(
        list(map(lambda conf: conf["h_index"], author_confs)))
    features['max_venue_h_index'] = np.max(
        list(map(lambda conf: conf["h_index"], author_confs)))
    features['min_venue_h_index'] = np.min(
        list(map(lambda conf: conf["h_index"], author_confs)))

    features['mean_venue_h_index_delta'] = np.mean(
        list(map(lambda conf: conf["h_index_delta"], author_confs)))
    features['max_venue_h_index_delta'] = np.max(
        list(map(lambda conf: conf["h_index_delta"], author_confs)))
    features['min_venue_h_index_delta'] = np.min(
        list(map(lambda conf: conf["h_index_delta"], author_confs)))

    features['mean_venue_citation_count'] = np.mean(
        list(map(lambda conf: conf["citation_count"], author_confs)))
    features['max_venue_citation_count'] = np.max(
        list(map(lambda conf: conf["citation_count"], author_confs)))
    features['min_venue_citation_count'] = np.min(
        list(map(lambda conf: conf["citation_count"], author_confs)))

    features['mean_venue_citation_delta'] = np.mean(
        list(map(lambda conf: conf["mean_citations_per_paper_delta"], author_confs)))
    features['max_venue_citation_delta'] = np.max(
        list(map(lambda conf: conf["mean_citations_per_paper_delta"], author_confs)))
    features['min_venue_citation_delta'] = np.min(
        list(map(lambda conf: conf["mean_citations_per_paper_delta"], author_confs)))

    # 提取年份特征（1997-2016 发的论文数量与引用次数）
    for year in range(1997, 2017):
        year_papers = list(
            filter(lambda paper: paper["year"] == year, author_papers))
        features[str(year) + "_cnt"] = len(year_papers)
        features[str(year) + "_citation"] = sum(list(
            map(lambda x: len(x["be_ref"]), year_papers)))
    # 根据期刊评价分数，计算在不同等级期刊发文的加权值
    for i in range(CLAZZ_NUM):
        features[str(i) + '_conf'] = 0.0
    for paper in author_papers:
        paper_conf_name = paper["conf"]
        if paper_conf_name in clazz_dict:
            paper_clazz_index = clazz_dict[paper_conf_name]
            features[str(paper_clazz_index) + "_conf"
                     ] += conf_rat_dict[paper_conf_name] * 1

    # 作为第一作者在各个年份发文的数量与被引数
    for year in range(1997, 2017):
        year_papers = list(
            filter(lambda paper: paper["year"] == year, first_author_papers))
        features["first_author_" + str(year) + "_cnt"] = len(year_papers)
        features["first_author_" + str(year) + "_citation"] = sum(list(
            map(lambda x: len(x["be_ref"]), year_papers)))
    # 作为第一作者在各个等级期刊发文的数量权重
    for i in range(CLAZZ_NUM):
        features['first_author_' + str(i) + '_conf'] = 0.0
    for paper in first_author_papers:
        paper_conf_name = paper["conf"]
        if paper_conf_name in clazz_dict:
            paper_clazz_index = clazz_dict[paper_conf_name]
            features["first_author_" + str(paper_clazz_index) + "_conf"
                     ] += conf_rat_dict[paper_conf_name] * 1

    # 合作作者的各种特征
    def cal_rank(_author_name=author_name, start_year=1936, end_year=2016):
        _paper_list = list(
            map(lambda p_index: papers[p_index], authors[author_name]["papers"]))
        _filter_paper_list = list(filter(
            lambda paper: paper["year"] >= start_year and paper["year"] <= end_year, _paper_list))
        if len(_filter_paper_list) == 0:
            return 0
        else:
            _paper_list.sort(key=lambda paper: paper["year"])
            _career_age = 2016 - _paper_list[0]["year"] + 1
            _range_career_age = _career_age - (2016 - end_year)
            _citation_ = sum(
                list(map(lambda _paper: len(_paper["be_ref"]), _filter_paper_list)))
            if len(_filter_paper_list) != 0:
                return _citation_ / len(_filter_paper_list) / _range_career_age
            else:
                return 0

    coauthors = list(map(lambda paper: paper["author"], author_papers))
    coauthors = list(reduce(lambda x, y: x + y, coauthors))
    coauthors.remove(author_name)
    if len(coauthors) != 0: 
        coauthors_rank = list(map(lambda _author_name: cal_rank(
            _author_name=_author_name), coauthors))
        features['mean_coauthor_rank'] = np.mean(coauthors_rank)
        features['max_coauthor_rank'] = np.max(coauthors_rank)
        features['min_coauthor_rank'] = np.min(coauthors_rank)

        coauthors_delta = list(map(lambda _author_name: cal_rank(
            _author_name=_author_name) - cal_rank(_author_name=_author_name, end_year=2014), coauthors))

        features['mean_coauthor_rank_delta'] = np.mean(coauthors_delta)
        features['max_coauthor_rank_delta'] = np.max(coauthors_delta)
        features['min_coauthor_rank_delta'] = np.min(coauthors_delta)
    else:
        features['mean_coauthor_rank'] = 0
        features['max_coauthor_rank'] = 0
        features['min_coauthor_rank'] = 0
        features['mean_coauthor_rank_delta'] = 0
        features['max_coauthor_rank_delta'] = 0
        features['min_coauthor_rank_delta'] = 0

    return ",".join([str(features[feature]) for feature in feature_head.split(",")]) + "\n"


conf_rat_dict = {}
clazz_list, clazz_dict = [], {}


def conf_rank():
    print("计算期刊评价分数")
    conf_rat_list = []
    for conf_name in confs.keys():
        conf_rat_list.append({conf_name: sum(list(map(lambda p_index: len(
            papers[p_index]["be_ref"]), confs[conf_name]["papers"]))) / len(confs[conf_name]["papers"])})

    conf_rat_list = list(
        filter(lambda r: list(r.values())[0] != 0, conf_rat_list))
    conf_rat_list.sort(key=lambda r: list(r.values())[0])
    each_clazz_num = int(len(conf_rat_list) / CLAZZ_NUM)

    for i in range(CLAZZ_NUM):
        clazz_list.append(
            conf_rat_list[i * each_clazz_num: (i + 1) * each_clazz_num])

    if each_clazz_num * CLAZZ_NUM < len(conf_rat_list):
        clazz_list[-1] += conf_rat_list[each_clazz_num * CLAZZ_NUM:]

    for clazz_index, clazz in enumerate(clazz_list):
        for j in clazz:
            clazz_dict[list(j.keys())[0]] = clazz_index

    for i in conf_rat_list:
        conf_rat_dict[list(i.keys())[0]] = list(i.values())[0]


conf_rank()
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
                break
    else:
        print("维数检查完毕")


check_dimension()
