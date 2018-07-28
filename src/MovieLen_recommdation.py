# coding=utf-8
from surprise import SVD,KNNWithMeans,SVDpp,NMF,BaselineOnly,KNNBaseline,KNNBasic,AlgoBase
from surprise import Dataset
from surprise import evaluate,print_perf
from surprise import Reader
from surprise import GridSearch
import os
import io

# # 加载文件所在路径
# file_path=os.path.expanduser(r'E:\JiangIntellijWorkingSpace\tools\music_recommendation\ml-100k\u.data')
# # 告诉reader,文本的格式是怎么样的
# reader=Reader(line_format='user item rating timestamp',sep='\t')
# data=Dataset.load_from_file(file_path,reader=reader)
# # #默认载入movielens数据集
# # data=Dataset.load_builtin('ml-100k')
# # k折交叉验证（k=5）
# data.split(n_folds=5)
# # 试一把SVD矩阵分解方法
# algo=KNNWithMeans()
# # 在数据集上测试一下效果
# perf=evaluate(algo,data,measures=['RMSE','MAE'])
# # 输出结果
# print_perf(perf)
#
# # 算法调参（让推荐系统有更好的效果）
#
# #定义好需要优选的参数网格
# para_grid={'n_epochs':[5,10],'lr_all':[0.002,0.005],'reg_all':[0.4,0.6],}
# grid_search=GridSearch(SVD,para_grid,measures={'RMSE','FCP'})
# grid_search.evaluate(data)
# # 输出调优的参数组
# # 输出最好的RMSE结果
# print(grid_search.best_score['RMSE'])
# print(grid_search.best_params['RMSE'])
# print(grid_search.best_score['FCP'])
# print(grid_search.best_params['FCP'])

def read_item_names():
    """
    获取电影名到电影id和电影id到电影名的映射
    :return:
    """
    # 加载文件所在路径
    file_path=os.path.expanduser(r'E:\JiangIntellijWorkingSpace\tools\music_recommendation\ml-100k\u.item')
    rid_to_name={}
    name_to_rid={}
    with io.open(file_path,'r',encoding='ISO-8859-1')as f:
        for line in f:
            line=line.split('|')
            """
            构造id字典
            """
            rid_to_name[line[0]]=line[1]
            name_to_rid[line[1]]=line[0]


    return  rid_to_name,name_to_rid
# 首先，用算法计算相互间的相似度
file_path=os.path.expanduser(r'E:\JiangIntellijWorkingSpace\tools\music_recommendation\ml-100k\u.data')
# 告诉reader,文本的格式是怎么样的
reader=Reader(line_format='user item rating timestamp',sep='\t')
data=Dataset.load_from_file(file_path,reader=reader)
trainset=data.build_full_trainset()
sim_options={'name':'pearson_baseline','user_based':False}
algo=KNNWithMeans(sim_options)
algo.train(trainset)
# 获取电影名到电影id和电影id到电影名的映射
rid_to_name,name_to_rid=read_item_names()
toy_story_raw_id=name_to_rid['Toy Story (1995)']
toy_story_inner_id=algo.trainset.to_inner_iid(toy_story_raw_id)
toy_story_neighbors=algo.get_neighbors(toy_story_inner_id,k=10)
toy_story_neighbors=(algo.trainset.to_raw_iid(inner_id)for inner_id in toy_story_neighbors)
toy_story_neighbors=(rid_to_name[rid]for rid in toy_story_neighbors)
print('the 10 nearest neighbors of it are(为你推荐最相近的10首电影):')
for movie in toy_story_neighbors:
    print movie



