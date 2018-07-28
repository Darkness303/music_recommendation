# coding=utf-8
from surprise import SVD,KNNWithMeans,SVDpp,NMF,BaselineOnly,KNNBaseline,KNNBasic,AlgoBase
from surprise import Dataset
from surprise import evaluate,print_perf
from surprise import Reader
from surprise import GridSearch
import os
import io
import cPickle as pickle
import sys
reload(sys)
sys.setdefaultencoding('utf-8')  #  上面三行是为了解决'ascii' codec can't encode characters in position 0-10: ordinal not in range(128)问题
import json
# 处理json文件函数
def preprocessing_Json():
    """加载json文件所在路径"""
    with open(r'E:\JiangIntellijWorkingSpace\tools\music_recommendation\preprocessing_naiveData.txt','w')as frr: #将大文件中的读取出来 放置到另个文件中
        frr.write("")
        print("文件初始化完成>>>")
    with open(r'E:\JiangHeSong\vm_share_file\playlistdetail.all.json', 'r') as f:
        i = 0

        while (i<100):# 由于原始爬取的json文件太大，采用取用一部分的数据
            i += 1
            print(u'正在载入第%s行......' % i)
            try:
                lines = f.readline()  # 使用逐行读取的方法
                review_text = json.loads(lines)  # 解析每一行数据
                name=review_text['result']['name']  # 把歌单的名字取出来
                tags=",".join(review_text['result']['tags'])  # 有多个标签，所以采用join方法
                subscribed_count=review_text['result']['subscribedCount']

                if(subscribed_count<50):
                    print "订阅数过小"
                playlist_id=review_text['result']['id']
                song_info=''
                songs=review_text['result']['tracks']
                for song in songs:
                    try:
                        song_info+="\t"+":::".join([str(song['id']),song['name'],song['artists'][0]['name'],str(song['popularity'])])

                    except Exception as e:
                        continue
                data=name+"##"+tags+"##"+str(playlist_id)+"##"+str(subscribed_count)+song_info+"\n"

                with open(r'E:\JiangIntellijWorkingSpace\tools\music_recommendation\preprocessing_naiveData.txt','a')as fr: #将大文件中的读取出来 放置到另个文件中
                    fr.write(data)
                    print("加载文件完成!!!")
            except Exception as e:
                print(str(e))
                break
def transform_music_data():
    with open(r'E:\JiangIntellijWorkingSpace\tools\music_recommendation\transform_playlist_song_rating.txt','wb')as fl1: #将大文件中的读取出来 放置到另个文件中
        fl1.write("")
    print("用于重复运行时的初始化操作，文档写入不会重复！！")
    try:
        with open(r'E:\JiangIntellijWorkingSpace\tools\music_recommendation\preprocessing_naiveData.txt','r')as fl:
            line=fl.readline()
            songs=line.split("\t")
            print line
            print songs[1]
            t=0
            for song in songs[1:]:
                t=t+1
                song_id=song.split(":::")[0]
                if t<len(songs)-1:
                    rating=song.split(":::")[3]+"\n"
                else:
                    rating=song.split(":::")[3]
                # print line.split("\t")[0].split("##")[2]
                data="\t".join([line.split("\t")[0].split("##")[2],song_id,rating])
                with open(r'E:\JiangIntellijWorkingSpace\tools\music_recommendation\transform_playlist_song_rating.txt','ab')as fl2: #将大文件中的读取出来 放置到另个文件中
                    fl2.write(data)

                    print("转换完成!!!")
    except Exception,e:
        print e
    fl2.close()
    fl1.close()
def read_item_names():
    """
    获取歌单名到歌单id和歌单id到歌单名的映射
    :return:
    格式为：歌单名##tags##歌单id##订阅数  。。。。。
    """

    file_path=os.path.expanduser(r'E:\JiangIntellijWorkingSpace\tools\music_recommendation\preprocessing_naiveData.txt')
    rid_to_name={}
    name_to_rid={}
    with io.open(file_path,'r',encoding='utf-8')as f:
        for line in f:
            line=line.split('\t')
            """
            构造id字典
            """
            rid_to_name[line[0].split("##")[2]]=line[0].split("##")[0]
            name_to_rid[line[0].split("##")[0]]=line[0].split("##")[2]
    return  rid_to_name,name_to_rid

def model_training_and_evalution():
    print "欢迎来到 训练阶段"
    file_path=os.path.expanduser(r'E:\JiangIntellijWorkingSpace\tools\music_recommendation\transform_playlist_song_rating.txt')
    reader=Reader(line_format='user item rating',sep='\t')
    music_data=Dataset.load_from_file(file_path,reader=reader)
    print("构建数据集")
    trainset=music_data.build_full_trainset()
    print"开始训练模型....."
    sim_options={'name':'pearson_baseline','user_based':False}
    algo=KNNWithMeans(sim_options)
    algo.train(trainset)
    rid_to_name,name_to_rid=read_item_names()
    # print name_to_rid
    toy_story_raw_id=name_to_rid[u'Over The Horizon-SAMSUNG GALAXY THEME']
    # toy_story_raw_id=423245641
    print toy_story_raw_id
    toy_story_inner_id=algo.trainset.to_inner_iid(toy_story_raw_id)
    toy_story_neighbors=algo.get_neighbors(toy_story_inner_id,k=10)
    toy_story_neighbors=(algo.trainset.to_raw_iid(inner_id)for inner_id in toy_story_neighbors)
    toy_story_neighbors=(rid_to_name[rid]for rid in toy_story_neighbors)
    print('the 10 nearest neighbors of it are(为你推荐最相近的10首歌单):')
    for music in toy_story_neighbors:
        print music

preprocessing_Json()
transform_music_data()
model_training_and_evalution()

