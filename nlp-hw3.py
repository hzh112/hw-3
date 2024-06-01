import os
import jieba
import re
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import cluster
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

def preprocess_text(text):
    """读取所有语料库信息，只保留中文字符，去除隐藏符号、标点符号等无用信息"""
    punctuation_pattern = r'[。，、；：？！（）《》【】“”‘’…—\-,.:;?!\[\](){}\'"<>]'
    text0 = re.sub(punctuation_pattern, '', text)
    text1 = re.sub(r'[\n\r\t]', '', text0)
    text2 = re.sub(r'[^\u4e00-\u9fa5]', '', text1)
    return text2

def load_stopwords(filepath):
    """读取停用词"""
    with open(filepath, 'r', encoding='utf-8') as file:
        stopwords = set([line.strip() for line in file.readlines()])
    return stopwords

def cut_text(text, stopwords):
    words = jieba.cut(text)
    return [word for word in words if word not in stopwords and word.strip()]

'''加载文本数据'''
corpus = []
directory = "C:\\Users\\Acer\\Desktop\\jyxstxtqj_downcc.com"
for filename in os.listdir(directory):
    if filename.endswith('.txt'):
        file_path = os.path.join(directory, filename)
        with open(file_path, 'r', encoding='ansi') as file:
            text = file.read()
            cleaned_text = preprocess_text(text)
            corpus.append(cleaned_text)

'''加载停用词列表.分字/词并过滤停用词'''
stopwords = load_stopwords("C:\\Users\\Acer\\Desktop\\cn_stopwords.txt")
processed_corpus = [cut_text(text, stopwords) for text in corpus]

'''训练Word2Vec模型 '''
model = Word2Vec(sentences=processed_corpus, vector_size=100, window=5, min_count=5, workers=16, epochs=50)

# 1
'''保存模型'''
model.save("jin_yong_word2vec.model")
word1 = "丐帮"
word2 = "华山派"
similarity_score = model.wv.similarity(word1, word2)
print(f"词语 '{word1}' 和 '{word2}' 的相似度得分为：{similarity_score}")

#2
model = Word2Vec.load("jin_yong_word2vec.model")

# 提取所有词的词向量
X = model.wv[model.wv.index_to_key]

# 使用KMeans算法进行聚类分析，设置聚类数量为10
num_clusters = 10
kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(X)
labels = kmeans.labels_

# 使用TSNE进行降维，以便在2D空间中可视化
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
X_tsne = tsne.fit_transform(X)

# 选择一个聚类中心进行展示，这里依旧以第0号聚类中心为例
cluster_id = 0
cluster_indices = [index for index, label in enumerate(labels) if label == cluster_id]
cluster_words = [model.wv.index_to_key[index] for index in cluster_indices]

# 可视化
plt.figure(figsize=(10, 5))
plt.scatter(X_tsne[cluster_indices, 0], X_tsne[cluster_indices, 1], c='red', label=f'Cluster {cluster_id}')
plt.legend()
plt.title('Word2Vec Model Word Clusters Visualization')

plt.show()

# 3
def paragraph_vector(model, text):
    """
    通过Word2Vec模型计算文本的平均向量。
    :param model: Word2Vec模型
    :param text: 文本，字符串格式
    :return: 平均词向量
    """
    # 分词并过滤掉没有在模型中的词
    words = [word for word in jieba.cut(text) if word not in stopwords]
    words = [word for word in words if word in model.wv.key_to_index]
    # 如果有有效的词，则计算平均词向量；如果没有，则返回None
    if words:
        return np.mean([model.wv[word] for word in words], axis=0)
    else:
        return None

def calculate_similarity(vec1, vec2):
    """
    计算两个向量之间的余弦相似度。
    :param vec1: 向量1
    :param vec2: 向量2
    :return: 余弦相似度值
    """
    # 将向量转换为适合cosine_similarity的形式
    vec1 = vec1.reshape(1, -1)
    vec2 = vec2.reshape(1, -1)
    return cosine_similarity(vec1, vec2)[0][0]

# 加载Word2Vec模型
model = Word2Vec.load("jin_yong_word2vec.model")

# 选择或生成文本段落
paragraph1 = "乔峰笑道：“好极，送了这两件利器给我！”双手抢起钢盾，盘旋飞舞。这两块钢盾当真是攻守俱臻佳妙的利器，只听得“啊唷”、“呵呵”几声惨呼，已有五人死在钢盾之下。"
paragraph2 = "任我行冷笑道：“是吗？因此你将我关在西湖湖底，教我不见天日。：东方不败道：”我没杀你，是不是？只须我叫江南四友不送水给你喝，你能捱得十天半月吗？“任我行道：”这样说来，你待我还算不错了？“东方不败道：”正是。我让你在杭州西湖颐养天年。常言道，上有天堂，下有苏杭。西湖风景，那是天下有名的了，孤山梅庄，更是西湖景色绝佳之处。”"

# 生成段落向量
vec1 = paragraph_vector(model, preprocess_text(paragraph1))
vec2 = paragraph_vector(model, preprocess_text(paragraph2))

# 如果两个段落向量都有效，计算它们的相似度
if vec1 is not None and vec2 is not None:
    similarity = calculate_similarity(vec1, vec2)
    print(f"段落1与段落2的相似度为：{similarity}")
else:
    print("无法计算相似度，因为至少有一个段落的所有词都不在Word2Vec模型的词汇表中。")