import numpy as np
import utils as Util


class DecisionTree():
    def __init__(self):
        self.clf_name = "DecisionTree"
        self.root_node = None

    def train(self, features, labels):
        # features: List[List[float]], labels: List[int]
        # init
        assert (len(features) > 0)
        num_cls = np.unique(labels).size

        # build the tree
        self.root_node = TreeNode(features, labels, num_cls)
        if self.root_node.splittable:
            self.root_node.split()

        return

    def predict(self, features):
        # features: List[List[any]]
        # return List[int]
        y_pred = []
        for idx, feature in enumerate(features):
            pred = self.root_node.predict(feature)
            y_pred.append(pred)
        return y_pred


class TreeNode(object):
    def __init__(self, features, labels, num_cls):
        # features: List[List[any]], labels: List[int], num_cls: int
        self.features = features
        self.labels = labels
        self.children = []
        self.num_cls = num_cls
        # find the most common labels in current node
        count_max = 0
        for label in np.unique(labels):
            if self.labels.count(label) > count_max:
                count_max = labels.count(label)
                self.cls_max = label
                # splitable is false when all features belongs to one class
        if len(np.unique(labels)) < 2:
            self.splittable = False
        else:
            self.splittable = True

        self.dim_split = None  # the index of the feature to be split

        self.feature_uniq_split = None  # the possible unique values of the feature to be split

    #TODO: try to split current node
    def split(self):
        if self.splittable==True:
            index_use=[]
            features=np.array(self.features)
            n=len(self.features[0])
            nn=len(self.features)
            bestIG=-1.0
            baseEn=0.0
            bestcc=0
            labelLength = len(self.labels)
            labelCount = len(set(self.labels))

            for i in range(labelCount):
                a = float(self.labels.count(list(set(self.labels))[i])) / float(labelLength)
                if a > 0:
                    baseEn += -(a * np.log2(a))
            for i in range(n):
                branchs=[]
                featureL=[aaa[i] for aaa in self.features]
                ind=list(set(featureL))
                cc=len(ind)
                for attr in ind:
                    l=[]
                    for j in range(nn):
                        if featureL[j]==attr:
                            l.append(self.labels[j])
                    d={}
                    for ii in l:
                        if ii not in d:
                            d[ii]=1
                        else:
                            d[ii]+=1
                    l1=[]
                    for iii in d.values():
                        l1.append(iii)
                    while len(l1)<self.num_cls:
                        l1.append(0)
                    branchs.append(l1)
                Info=Util.Information_Gain(baseEn,branchs)
            
                if Info - bestIG > 1e-5:
                    bestcc=cc
                    bestIG=Info
                    self.dim_split=i
                    self.feature_uniq_split=np.unique(features[:,self.dim_split]).tolist()
                elif Info==bestIG:
                    if cc>bestcc:
                        bestcc=cc
                        self.dim_split=i
                        self.feature_uniq_split=np.unique(features[:,self.dim_split]).tolist()
            if self.features==None:
                self.splittable=False
                return
            if bestIG < 1e-5:
                self.splittable=False
                return

            if self.num_cls==1:
                self.splittable=False
                return
        
            if self.feature_uniq_split==None:
                self.splittable=False
                return
        
            index_use.append(i)
            labels = self.labels
            t = self.feature_uniq_split
            c = self.dim_split
            for m in t:
                res = []
                l = []
                for i in range(len(features)):

                    if m == features[i][c]:
                        l.append(labels[i])
                        a = list(features[i])
                        a.remove(m)
                        res.append(a)
                num_cls=len(set(l))

                child = TreeNode(res, l, num_cls)
                if len(index_use)==n:
                    child.splittable=False
                if res==None:
                    child.splittable=False
                if len(set(l))==1:
                    child.splittable=False
                self.children.append(child)

            for child in self.children:
                if child.splittable:
                    child.split()
            return
        else:
            
            return
  
        
    # TODO: predict the branch or the class
    def predict(self, feature):
        if self.splittable:
            if self.dim_split is not None:
                if self.num_cls==1:
                    return self.cls_max
                if self.dim_split>=len(feature):
                    return self.cls_max
                a=feature[self.dim_split]
                res=[]
                if len(feature)>=2:
                    for i in range(len(feature)):
                        if i != self.dim_split:
                            res.append(feature[i])
                    feature=res
                    if a in self.feature_uniq_split:
                        
                        return self.children[self.feature_uniq_split.index(a)].predict(feature)
                    else:
                        self.splittable=False
                        return self.cls_max
                        
                else:
                    
                    self.splittable=False
                    return self.cls_max  
            else:
                self.splittable=False
                return self.cls_max
        else:
            self.splittable=False
            return self.cls_max

