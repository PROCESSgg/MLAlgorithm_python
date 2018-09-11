import math
import operator


# 计算信息熵
def entropy(dataset):
    sample_size = len(dataset)
    entropy_d = 0
    d = {}
    for sample in dataset:
        class_ = sample[-1]
        if class_ not in d.keys():
            d[class_] = 0
        d[class_] += 1
    for class_ in d.keys():
        prob = float(d[class_]) / sample_size
        entropy_d -= prob * math.log(prob, 2)
    return entropy_d

#计算IV(a)
def IV(dataset, attrlist):
    d = {}
    IVa = 0
    for attr in attrlist:
        if attr not in d.keys():
            d[attr] = 0
        d[attr] += 1
    for attr in d.keys():
        prob = float(d[attr])/len(dataset)
        IVa -=  prob* math.log(prob, 2)
    return IVa

# 针对某个特征的属性进行划分
def attr_part(dataset, feature_number, attribute):
    attr_part_res = []
    for sample in dataset:
        if sample[feature_number] == attribute:
            attr_part_res.append(sample[:feature_number] + sample[feature_number + 1:])
    return attr_part_res


# 选择最大信息增益特征进行划分
def chooseBestFeatureToSplit(dataset):
    num_features = len(dataset[0]) - 1
    baseEntropy = entropy(dataset)
    bestInfoGain = 0
    bestFeature = -1
    for i in range(num_features):
        feature_entropy = 0
        attr_list = [sample[i] for sample in dataset]

        attr_set = set(attr_list)
        for attr in attr_set:
            attr_part_res = attr_part(dataset, i, attr)
            prob = len(attr_part_res) / float(len(dataset))
            feature_entropy += prob * entropy(attr_part_res)
        if (baseEntropy - feature_entropy)/IV(dataset,attr_list) > bestInfoGain:
            bestInfoGain = (baseEntropy - feature_entropy)/IV(dataset,attr_list)
            bestFeature = i

    return bestFeature


# 计算数据集中最多的类别
def major_class(dataset):
    class_list = [sample[-1] for sample in dataset]
    count_d = {}
    for class_ in class_list:
        if class_ not in count_d.keys():
            count_d[class_] = 0
        count_d[class_] += 1
    sortedcount_d = sorted(count_d.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedcount_d[0][0]


# 创建决策树
'''
　　　　输入：训练数据集D，特征集A，阈值ε（预剪枝用，后剪枝不需要此项）；
　　　　输出：决策树T。
　　（1）若D中所有样本属于同一类Ck,则T为但结点树，并将Ck作为该结点的类标记，返回T；
　　（2）若A = Ø,则T为单结点树，并将D中样本数最多的类Ck作为该结点的类标记，返回T；
　　（3）否则，计算A中各个特征对D的信息增益或者信息增益比，选择信息增益或信息增益比最大的特征 Ag；
　　（4）如果 Ag 的信息增益或信息增益比小于阈值ε，则置T为但结点树，并将D中样本数最多的类Ck作为该结点的类标记，返回T；（后剪枝没有这步）
　　（5）如果Ag的每一种可能值ai，依 Ag = ai 将D分割为若干非空子集Di，将Di中样本数最多的类作为标记，构建子结点，由结点及其子结点构成树T，返回T；
　　（6）对第i个子结点，以Di为训练集，以A - { Ag } 为特征集，递归地调用步骤（1）~（5），得到子树Ti，返回Ti 。
'''


def create_id3_tree(dataset, labels):
    class_list = [sample[-1] for sample in dataset]
    # (1)
    if len(set(class_list)) == 1:
        return class_list[0]
    # (2)
    if len(labels) == 0:
        return major_class(dataset)
    # (3)
    bestFeature_num = chooseBestFeatureToSplit(dataset)
    bestFeature = labels[bestFeature_num]
    myTree = {bestFeature: {}}
    del (labels[bestFeature_num])
    # (6)
    attributes = [sample[bestFeature_num] for sample in dataset]
    for attr in attributes:
        sublabels = labels[:]
        myTree[bestFeature][attr] = create_id3_tree(attr_part(dataset, bestFeature_num, attr), sublabels)
    return myTree


def CreateDataSet():
    dataset = [['SUNNY', 'HOT', 'HIGH', 'WEAK', 'n'],
               ['SUNNY', 'HOT', 'HIGH', 'STRONG', 'n'],
               ['OVERCAST', 'HOT', 'HIGH', 'WEAK', 'y'],
               ['RAIN', 'MILD', 'HIGH', 'WEAK', 'y'],
               ['RAIN', 'COOL', 'NORMAL', 'WEAK', 'y'],
               ['RAIN', 'COOL', 'NORMAL', 'STRONG', 'n'],
               ['OVERCAST', 'COOL', 'NORMAL', 'STRONG', 'y'],
               ['SUNNY', 'MILD', 'HIGH', 'WEAK', 'n'],
               ['SUNNY', 'COOL', 'NORMAL', 'WEAK', 'y'],
               ['RAIN', 'MILD', 'NORMAL', 'WEAK', 'y'],
               ['SUNNY', 'MILD', 'NORMAL', 'STRONG', 'y'],
               ['OVERCAST', 'MILD', 'HIGH', 'STRONG', 'y'],
               ['OVERCAST', 'HOT', 'NORMAL', 'WEAK', 'y'],
               ['RAIN', 'MILD', 'HIGH', 'STRONG', 'n']]
    labels = ['outlook', 'temp', 'hum', 'windy']
    return dataset, labels


if __name__ == '__main__':
    myDat, labels = CreateDataSet()
    print(create_id3_tree(myDat, labels))