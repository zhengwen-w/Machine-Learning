import numpy as np
from knn import KNN

############################################################################
# DO NOT MODIFY ABOVE CODES
############################################################################


# TODO: implement F1 score
def f1_score(real_labels, predicted_labels):
    n=len(real_labels)
    f1a=0
    f1b=0
    for i in range(n):
        f1a+=2*real_labels[i]*predicted_labels[i]
        f1b+=real_labels[i]+predicted_labels[i]

    return f1a/f1b
    """
    Information on F1 score - https://en.wikipedia.org/wiki/F1_score
    :param real_labels: List[int]
    :param predicted_labels: List[int]
    :return: float
    """
    assert len(real_labels) == len(predicted_labels)
    

class Distances:
    @staticmethod
    # TODO
    def minkowski_distance(point1, point2):
        x=np.array(point1)
        y=np.array(point2)
        dist=abs(x-y)**3
        distSum = 0
        for i in dist:
            distSum+=i
        distance=pow(distSum,1/3)
        return distance
        
        """
        Minkowski distance is the generalized version of Euclidean Distance
        It is also know as L-p norm (where p>=1) that you have studied in class
        For our assignment we need to take p=3
        Information on Minkowski distance - https://en.wikipedia.org/wiki/Minkowski_distance
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        

    @staticmethod
    # TODO
    def euclidean_distance(point1, point2):
        x=np.array(point1)
        y=np.array(point2)
        dist=(x-y)**2
        distSum = 0
        for i in dist:
            distSum+=i
        distance=distSum**0.5
        return distance
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        

    @staticmethod
    # TODO
    def inner_product_distance(point1, point2):
        x=np.array(point1)
        y=np.array(point2)
        return np.inner(x,y)
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        

    @staticmethod
    # TODO
    def cosine_similarity_distance(point1, point2):
        x=np.array(point1)
        y=np.array(point2)
        s1=np.inner(x,y)
        temp1=0
        temp2=0
        for i in range(len(point1)):
            temp1+=point1[i]**2
            temp2+=point2[i]**2
        s2=temp1**0.5*temp2**0.5
        return 1-s1/s2
        """
       :param point1: List[float]
       :param point2: List[float]
       :return: float
       """
       

    @staticmethod
    # TODO
    def gaussian_kernel_distance(point1, point2):
        x=np.array(point1)
        y=np.array(point2)
        dist=(x-y)**2
        distSum = 0
        for i in dist:
            distSum+=i
        distSum=distSum*(-0.5)
        distance=-np.exp(distSum)
        return distance
        """
       :param point1: List[float]
       :param point2: List[float]
       :return: float
       """
        


class HyperparameterTuner:
    def __init__(self):
        self.best_k = None
        self.best_distance_function = None
        self.best_scaler = None
        self.best_model = None

    # TODO: find parameters with the best f1 score on validation dataset
    def tuning_without_scaling(self, distance_funcs, x_train, y_train, x_val, y_val):
        best_score=0
        best_k=-1
        best_dist=''
        best_model=None
        for key in distance_funcs:
            for k in range(1,30,2):
                if k>len(x_train):
                    break
                a=KNN(k,distance_funcs[key])
                a.train(x_train,y_train)
                P=a.predict(x_val)
                score=f1_score(y_val,P)
                if score>best_score:
                    best_score=score
                    best_k=k
                    best_dist=key
                    best_model=a
        self.best_k = best_k
        self.best_distance_function =best_dist
        self.best_model = best_model
        """
        In this part, you should try different distance function you implemented in part 1.1, and find the best k.
        Use k range from 1 to 30 and increment by 2. Use f1-score to compare different models.

        :param distance_funcs: dictionary of distance functions you must use to calculate the distance.
            Make sure you loop over all distance functions for each data point and each k value.
            You can refer to test.py file to see the format in which these functions will be
            passed by the grading script
        :param x_train: List[List[int]] training data set to train your KNN model
        :param y_train: List[int] train labels to train your KNN model
        :param x_val:  List[List[int]] Validation data set will be used on your KNN predict function to produce
            predicted labels and tune k and distance function.
        :param y_val: List[int] validation labels

        Find(tune) best k, distance_function and model (an instance of KNN) and assign to self.best_k,
        self.best_distance_function and self.best_model respectively.
        NOTE: self.best_scaler will be None

        NOTE: When there is a tie, choose model based on the following priorities:
        Then check distance function  [euclidean > minkowski > gaussian > inner_prod > cosine_dist]
        If they have same distance fuction, choose model which has a less k.
        """
        
        # You need to assign the final values to these variables
        self.best_k = best_k
        self.best_distance_function =best_dist
        self.best_model = best_model
       

    # TODO: find parameters with the best f1 score on validation dataset, with normalized data
    def tuning_with_scaling(self, distance_funcs, scaling_classes, x_train, y_train, x_val, y_val):
        best_score=-100
        best_k=None
        best_dist=''
        best_model=None
        best_scaler=None
        xtrain=x_train
        xval=x_val
        for key in distance_funcs:
            for i in scaling_classes:
                for k in range(1,30,2):
                    x_train=xtrain
                    x_val=xval
                    if k>len(x_train):
                        break
                    b=scaling_classes[i]()
                    x_train=b.__call__(x_train)
                    x_val=b.__call__(x_val)
                    a=KNN(k,distance_funcs[key])
                    a.train(x_train,y_train)
                    P=a.predict(x_val)
                    score=f1_score(y_val,P)
                    if score>best_score:
                        best_score=score
                        best_k=k
                        best_scaler=i
                        best_dist=key
                        best_model=a
        self.best_scaler=best_scaler
        self.best_k = best_k
        self.best_distance_function =best_dist
        self.best_model = best_model
        """
        This part is similar to Part 1.3 except that before passing your training and validation data to KNN model to
        tune k and disrance function, you need to create the normalized data using these two scalers to transform your
        data, both training and validation. Again, we will use f1-score to compare different models.
        Here we have 3 hyperparameters i.e. k, distance_function and scaler.

        :param distance_funcs: dictionary of distance funtions you use to calculate the distance. Make sure you
            loop over all distance function for each data point and each k value.
            You can refer to test.py file to see the format in which these functions will be
            passed by the grading script
        :param scaling_classes: dictionary of scalers you will use to normalized your data.
        Refer to test.py file to check the format.
        :param x_train: List[List[int]] training data set to train your KNN model
        :param y_train: List[int] train labels to train your KNN model
        :param x_val: List[List[int]] validation data set you will use on your KNN predict function to produce predicted
            labels and tune your k, distance function and scaler.
        :param y_val: List[int] validation labels

        Find(tune) best k, distance_funtion, scaler and model (an instance of KNN) and assign to self.best_k,
        self.best_distance_function, self.best_scaler and self.best_model respectively

        NOTE: When there is a tie, choose model based on the following priorities:
        For normalization, [min_max_scale > normalize];
        Then check distance function  [euclidean > minkowski > gaussian > inner_prod > cosine_dist]
        If they have same distance function, choose model which has a less k.
        """
        
        # You need to assign the final values to these variables



class NormalizationScaler:
    def __init__(self):
        pass

    # TODO: normalize data
    def __call__(self, features):
        features=np.array(features)
        a=features*features
        b=a.sum(axis=1)
        res=[]
        for i in range(len(features)):
            res.append([])
            for j in features[i]:
                if b[i]!=0:
                    res[i].append(j/b[i]**0.5)
                else:
                    res[i].append(0)
        return res
        
        """
        Normalize features for every sample

        Example
        features = [[3, 4], [1, -1], [0, 0]]
        return [[0.6, 0.8], [0.707107, -0.707107], [0, 0]]

        :param features: List[List[float]]
        :return: List[List[float]]
        """


class MinMaxScaler:
    """
    Please follow this link to know more about min max scaling
    https://en.wikipedia.org/wiki/Feature_scaling
    You should keep some states inside the object.
    You can assume that the parameter of the first __call__
    will be the training set.

    Hints:
        1. Use a variable to check for first __call__ and only compute
            and store min/max in that case.

    Note:
        1. You may assume the parameters are valid when __call__
            is being called the first time (you can find min and max).

    Example:
        train_features = [[0, 10], [2, 0]]
        test_features = [[20, 1]]

        scaler1 = MinMaxScale()
        train_features_scaled = scaler1(train_features)
        # train_features_scaled should be equal to [[0, 1], [1, 0]]

        test_features_scaled = scaler1(test_features)
        # test_features_scaled should be equal to [[10, 0.1]]

        new_scaler = MinMaxScale() # creating a new scaler
        _ = new_scaler([[1, 1], [0, 0]]) # new trainfeatures
        test_features_scaled = new_scaler(test_features)
        # now test_features_scaled should be [[20, 1]]

    """

    def __init__(self):
        self.minval=None
        self.maxval=None

    def __call__(self, features):
        features=np.array(features)
        if self.minval is None:
            self.minval=features.min(0)
        if self.maxval is None:
            self.maxval=features.max(0)
        normData=features-np.tile(self.minval,(features.shape[0],1))
        normData=normData/np.tile(self.maxval-self.minval,(features.shape[0],1))
        return normData
    
        """
        normalize the feature vector for each sample . For example,
        if the input features = [[2, -1], [-1, 5], [0, 0]],
        the output should be [[1, 0], [0, 1], [0.333333, 0.16667]]

        :param features: List[List[float]]
        :return: List[List[float]]
        """
   
