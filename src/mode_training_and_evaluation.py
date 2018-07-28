# coding=utf-8
from surprise import SVD,KNNWithMeans
from surprise import Dataset
from surprise import evaluate,print_perf
from surprise import Reader
from surprise import GridSearch
import os
import io

# 加载文件所在路径
file_path=os.path.expanduser(r'E:\JiangIntellijWorkingSpace\tools\music_recommendation\ml-100k\u.data')
# 告诉reader,文本的格式是怎么样的
reader=Reader(line_format='user item rating timestamp',sep='\t')
data=Dataset.load_from_file(file_path,reader=reader)
# #默认载入movielens数据集
# data=Dataset.load_builtin('ml-100k')
# k折交叉验证（k=5）
data.split(n_folds=5)
# 试一把SVD矩阵分解方法
algo=SVD()
# 在数据集上测试一下效果
perf=evaluate(algo,data,measures=['RMSE','MAE'])
# 输出结果
print_perf(perf)

# 算法调参（让推荐系统有更好的效果）

#定义好需要优选的参数网格
para_grid={'n_epochs':[5,10],'lr_all':[0.002,0.005],'reg_all':[0.4,0.6],}
grid_search=GridSearch(SVD,para_grid,measures={'RMSE','FCP'})
grid_search.evaluate(data)
# 输出调优的参数组
# 输出最好的RMSE结果
print(grid_search.best_score['RMSE'])
print(grid_search.best_params['RMSE'])
print(grid_search.best_score['FCP'])
print(grid_search.best_params['FCP'])



