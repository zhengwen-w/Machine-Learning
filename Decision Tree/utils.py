import numpy as np
import hw1_dt as hw


# TODO: Information Gain function
def Information_Gain(S, branches):
    conditionEntropy=0
    prod=[]
    for l in branches:
        summ=sum(l)
        aaa=0
        for i in range(len(l)):
            if summ==0:
                break
            if l[i]==0:
                aaa+=0
            else:
                aaa+=-(float(l[i])/float(summ)*np.log2(float(l[i])/float(summ)))

        prod.append(aaa)

    branches=np.array(branches)
    prob=branches.sum(axis=1)

    s=sum(prob)
    for i in range(len(prob)):
        if prob[i]==0:
            conditionEntropy+=0
        else:
            conditionEntropy+=float(prob[i])/float(s)*prod[i]
    return S-conditionEntropy

# TODO: implement reduced error prunning function, pruning your tree on this function
def reduced_error_prunning(decisionTree, X_test, y_test):

    if not decisionTree.root_node.splittable:
        return
    
    """
    predict = node.predict()
    """
    
    if decisionTree.root_node.splittable:
        labels =y_test
        to_split = decisionTree.root_node.feature_uniq_split
        cut = decisionTree.root_node.dim_split
        dr=[]
        dl=[]

        for m in to_split:
            res = []
            l = []
            for i in range(len(X_test)):
                if m == X_test[i][cut]:
                    l.append(labels[i])
                    a = list(X_test[i])
                    a.remove(m)
                    res.append(a)
            dl.append(l)
            dr.append(res)
        for i in range(len(decisionTree.root_node.children)):
            a=hw.DecisionTree()
            a.root_node = decisionTree.root_node.children[i]
            if i<=(len(dr)-1) and i<=(len(dl)-1):
                reduced_error_prunning(a,dr[i],dl[i])
                error1=0
                error2=0
                if a.predict(dr[i])and dl[i]:
                    for x in range(len(a.predict(dr[i]))):
                        if a.predict(dr[i])[x]==dl[i][x]:
                            error1+=1
                    for y in dl[i]:
                        if y==a.root_node.cls_max:
                            error2+=1
                    if error1<=error2:
                        a.root_node.splittable=False
                        a.root_node.children=[]
                        a.root_node.feature_uniq_split=None
                        a.root_node.dim_split=None
                    
              
                else:
                    a.root_node.splittable=False
                    a.root_node.children=[]
                    a.root_node.feature_uniq_split=None
                    a.root_node.dim_split=None
            else:
                a.root_node.splittable=False
                a.root_node.children=[]
                a.root_node.feature_uniq_split=None
                a.root_node.dim_split=None
                 
    else:
        decisionTree.splittable=False
        decisionTree.children=[]
        decisionTree.feature_uniq_split=None
        decisionTree.dim_split=None
        return



# print current tree
def print_tree(decisionTree, node=None, name='branch 0', indent='', deep=0):
    if node is None:
        node = decisionTree.root_node
    print(name + '{')

    print(indent + '\tdeep: ' + str(deep))
    string = ''
    label_uniq = np.unique(node.labels).tolist()
    for label in label_uniq:
        string += str(node.labels.count(label)) + ' : '
    print(indent + '\tnum of samples for each class: ' + string[:-2])

    if node.splittable:
        print(indent + '\tsplit by dim {:d}'.format(node.dim_split))
        for idx_child, child in enumerate(node.children):
            print_tree(decisionTree, node=child, name='\t' + name + '->' + str(idx_child), indent=indent + '\t', deep=deep+1)
    else:
        print(indent + '\tclass:', node.cls_max)
    print(indent + '}')
