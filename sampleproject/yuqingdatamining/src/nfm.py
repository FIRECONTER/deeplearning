# -*- coding:utf-8 -*-
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import NMF, LatentDirichletAllocation

#第2步：定义一个展示主题和主题词的函数，用来展示结果
def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print ("Topic %d:" % (topic_idx))
        print (" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))

#第3步：导入数据，设定特征数
dataset = open('../data/公开信分词.txt',encoding='utf-8').readlines()
documents = dataset
no_features = 1000


#第4步：抽取NMF模型所需的tf-idf 特征
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(documents)
tfidf_feature_names = tfidf_vectorizer.get_feature_names()


#第5步：运行NMF模型和LDA模型
# 运行 NMF 模型
no_topics = 20
nmf = NMF(n_components=no_topics, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(tfidf)

#第6步：设定主题数，可以反复测试，哪个主题数比较合适
no_top_words = 15


#第7步：展示NMF模型的主题结果
display_topics(nmf, tfidf_feature_names, no_top_words)
