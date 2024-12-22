# -此仓库为机器学习期末作业
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
# （1）加载鸢尾花数据集
iris = load_iris()
X = iris.data  # 特征数据，是一个二维数组，每行代表一个样本的特征向量，这里有4个特征（属性）
y = iris.target  # 目标类别，是一维数组，对应每个样本的所属类别，共3类

# （2）划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# （3）创建高斯朴素贝叶斯分类器对象
gnb = GaussianNB()
# （4）使用训练数据进行训练
gnb.fit(X_train, y_train)
# （5）在测试集上进行预测
y_pred = gnb.predict(X_test)
# （6）计算准确率
accuracy = accuracy_score(y_test, y_pred)
print("准确率:", accuracy)




import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

# 模拟一些交易数据，这里是一个二维列表，每个子列表代表一次购物交易的商品列表
data = [['牛奶', '面包', '尿布'],
        ['可乐', '面包', '尿布', '啤酒'],
        ['牛奶', '尿布', '啤酒', '鸡蛋'],
        ['面包', '牛奶', '尿布', '啤酒'],
        ['面包', '牛奶', '尿布']]

# 使用TransactionEncoder对数据进行编码，转化为适合算法处理的格式（0-1矩阵形式）
te = TransactionEncoder()
te_ary = te.fit(data).transform(data)

# 将结果转换为Pandas DataFrame
df = pd.DataFrame(te_ary, columns=te.columns_)

# 应用Apriori算法挖掘频繁项集，设置最小支持度为0.3（即出现频率至少为30%）
frequent_itemsets = apriori(df, min_support=0.3, use_colnames=True)

# 根据频繁项集生成关联规则，设置最小置信度为0.7
# 添加 num_itemsets 参数，设置为2（假设我们只关心二项集的规则）
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7, num_itemsets=2)

print("频繁项集：")
print(frequent_itemsets)
print("关联规则：")
print(rules)




import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# （1）生成一些模拟的二维数据点，这里简单生成了3个簇的数据
np.random.seed(42)
data = np.vstack((np.random.randn(100, 2) + [2, 2],
                  np.random.randn(100, 2) + [-2, -2],
                  np.random.randn(100, 2) + [5, -5]))

# （2）创建K-Means聚类器对象，指定聚类的簇数为3
kmeans = KMeans(n_clusters=3, random_state=42)

# （3）对数据进行聚类训练
kmeans.fit(data)

# （4）获取聚类后的标签（每个数据点所属的簇标记）
labels = kmeans.labels_

# （5）获取聚类中心的坐标
centroids = kmeans.cluster_centers_

# （6）可视化聚类结果
plt.scatter(data[:, 0], data[:, 1], c=labels)
plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=200, c='r')
plt.title("K-Means Clustering Result")
plt.show()



(3)	import pandas as pd
import numpy as np

# 定义Wine数据集的列名
columns = ['Class', 'Alcohol', 'Malic_Acid', 'Ash', 'Alcalinity_of_Ash',
           'Magnesium', 'Total_Phenols', 'Flavanoids', 'Nonflavanoid_Phenols',
           'Proanthocyanins', 'Color_Intensity', 'Hue', 'OD280_OD315_of_Diluted_Wines', 'Proline']

# 读取数据
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data'
data = pd.read_csv(url, header=None, names=columns)

# 显示数据的前几行
print(data.head())


# 信息熵的计算函数
def entropy(data):
    values = data.iloc[:, 0].value_counts()  # 获取标签列
    total = len(data)
    entropy_value = 0
    for count in values:
        prob = count / total
        entropy_value -= prob * np.log2(prob)
    return entropy_value


# 信息增益的计算函数
def information_gain(data, attribute):
    attribute_index = data.columns.get_loc(attribute)  # 获取列的索引
    values = data.iloc[:, attribute_index].value_counts()  # 获取属性的所有不同取值
    entropy_before = entropy(data)  # 计算划分前的数据熵
    entropy_after = 0
    total_count = len(data)

    # 遍历该属性的每个取值，计算该属性的条件熵
    for value, count in values.items():
        subset = data[data.iloc[:, attribute_index] == value]
        subset_entropy = entropy(subset)
        entropy_after += (count / total_count) * subset_entropy

    return entropy_before - entropy_after


# ID3算法的实现
def id3(data, attributes):
    if len(data.iloc[:, 0].unique()) == 1:  # 只有一个类别
        return data.iloc[0, 0]

    if len(attributes) == 0:  # 所有属性都已经用完
        return data.iloc[:, 0].mode()[0]

    # 计算每个属性的信息增益
    gains = [information_gain(data, attr) for attr in attributes]
    best_attribute = attributes[np.argmax(gains)]

    tree = {best_attribute: {}}

    # 对于每个属性的取值，递归构建子树
    attribute_index = data.columns.get_loc(best_attribute)
    for value in data.iloc[:, attribute_index].unique():
        subset = data[data.iloc[:, attribute_index] == value]
        remaining_attributes = [attr for attr in attributes if attr != best_attribute]
        tree[best_attribute][value] = id3(subset, remaining_attributes)

    return tree


# 使用ID3算法构建决策树
attributes = columns[1:]  # 除去标签列
tree = id3(data, attributes)

# 输出决策树
print(tree)



(3)	import numpy as np
import pandas as pd
from collections import Counter


# 计算欧几里得距离
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


# KNN算法实现
def knn(train_data, train_labels, test_data, k):
    predictions = []
    for test_point in test_data:
        distances = []
        # 计算每个训练样本与测试样本的距离
        for i, train_point in enumerate(train_data):
            dist = euclidean_distance(test_point, train_point)
            distances.append((dist, train_labels[i]))
        distances.sort(key=lambda x: x[0])  # 按距离排序
        top_k = [label for _, label in distances[:k]]  # 选取前k个邻居
        most_common = Counter(top_k).most_common(1)  # 投票机制，选择出现最多的标签
        predictions.append(most_common[0][0])
    return predictions


# 读取Iris数据集
def load_iris_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
    data = pd.read_csv(url, header=None, names=columns)

    # 将数据集分为训练集和测试集
    data = data.sample(frac=1, random_state=42)  # 打乱数据
    train_data = data.iloc[:120, :-1].values  # 前120个作为训练集
    test_data = data.iloc[120:, :-1].values  # 后30个作为测试集
    train_labels = data.iloc[:120, -1].values
    test_labels = data.iloc[120:, -1].values

    return train_data, train_labels, test_data, test_labels


# 测试KNN算法
train_data, train_labels, test_data, test_labels = load_iris_data()
k = 3
predictions = knn(train_data, train_labels, test_data, k)

# 输出预测结果
print("Predictions:", predictions)
print("True Labels:", test_labels)
