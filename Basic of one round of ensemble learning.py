import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, \
    HistGradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB, BernoulliNB
# Import some data to play with
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC


class Ensemble:
    def __init__(self):
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None

    def load_data(self, text, train):
        s = np.loadtxt(text, dtype=np.float32, delimiter=' ')
        end = s.shape[1] - 1
        text_X = s[:, :end]
        text_y = s[:, -1]
        s = np.loadtxt(train, dtype=np.float32, delimiter=' ')
        end = s.shape[1] - 1
        train_X = s[:, :end]
        train_y = s[:, -1]
        self.x_train, self.x_test, self.y_train, self.y_test = train_X, text_X, train_y, text_y

    @staticmethod
    def __Classifiers__(name=None):
        # See for reproducibility
        random_state = 100
        kernel = 1.0 * RBF(1.0)
        if name == 'Neighbors':
            return RadiusNeighborsClassifier(radius=1.0)
        if name == 'Gaussian_Process':
            return GaussianProcessClassifier(kernel=kernel, random_state=random_state)
        if name == 'Gaussian_NB':
            return GaussianNB()
        if name == 'Bernoulli_NB':
            return BernoulliNB()
        if name == 'DecisionTree':
            return tree.DecisionTreeClassifier()
        if name == 'Bagging':
            return BaggingClassifier(base_estimator=SVC())
        if name == 'RandomForest':
            return RandomForestClassifier(n_estimators=10)
        if name == 'AdaBoost':
            return AdaBoostClassifier(n_estimators=100, random_state=random_state)
        if name == 'GradientBoosting':
            return GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1,
                                              random_state=random_state)
        if name == 'HistGradientBoosting':
            return HistGradientBoostingClassifier()
        if name == 'MLP':
            return MLPClassifier(random_state=random_state)

    # 1.6.2
    def __Neighbors__(self):
        # Decision Tree Classifier
        neigh = Ensemble.__Classifiers__(name='Neighbors')
        # Init Grid Search
        neigh.fit(self.x_train, self.y_train)

    # 1.7.2
    def __GPC__(self):
        # Decision Tree Classifier
        GPC = Ensemble.__Classifiers__(name='Gaussian_Process')
        # Init Grid Search
        GPC.fit(self.x_train, self.y_train)

    # 1.9.1
    def __Gaussian_NB__(self):
        # Decision Tree Classifier
        gnb = Ensemble.__Classifiers__(name='Gaussian_NB')
        # Init Grid Search
        gnb.fit(self.x_train, self.y_train)

    # 1.9.4
    def __Bernoulli_NB__(self):
        # Decision Tree Classifier
        bnb = Ensemble.__Classifiers__(name='Bernoulli_NB')
        # Init Grid Search
        bnb.fit(self.x_train, self.y_train)

    # 1.10.1
    def __DecisionTree__(self):
        # Decision Tree Classifier
        dt = Ensemble.__Classifiers__(name='DecisionTree')
        # Init Grid Search
        dt.fit(self.x_train, self.y_train)

    # 1.11.1
    def __Bagging__(self):
        # Decision Tree Classifier
        bag = Ensemble.__Classifiers__(name='Bagging')
        # Init Grid Search
        bag.fit(self.x_train, self.y_train)

    # 1.11.2
    def __RandomForest__(self):
        # Decision Tree Classifier
        Forest = Ensemble.__Classifiers__(name='RandomForest')
        # Init Grid Search
        Forest.fit(self.x_train, self.y_train)

    # 1.11.3
    def __AdaBoost__(self):
        # Decision Tree Classifier
        AdaBoost = Ensemble.__Classifiers__(name='AdaBoost')
        # Init Grid Search
        AdaBoost.fit(self.x_train, self.y_train)

    # 1.11.4
    def __GradientBoosting__(self):
        # Decision Tree Classifier
        Gdbt = Ensemble.__Classifiers__(name='GradientBoosting')
        # Init Grid Search
        Gdbt.fit(self.x_train, self.y_train)

    # 1.11.5
    def __HistGradientBoosting__(self):
        # Decision Tree Classifier
        HGdbt = Ensemble.__Classifiers__(name='HistGradientBoosting')
        # Init Grid Search
        HGdbt.fit(self.x_train, self.y_train)

    # 1.17.2
    def __MLPClassifier_1__(self):
        # Decision Tree Classifier
        MLP = Ensemble.__Classifiers__(name='MLP')
        # Init Grid Search
        MLP.fit(self.x_train, self.y_train)

    def __VotingClassifier__(self, fnameresult1):

        # Instantiate classifiers 实例化分类器

        # Neigh = Ensemble.__Classifiers__(name='Neighbors')
        GPC = Ensemble.__Classifiers__(name='Gaussian_Process')
        gnb = Ensemble.__Classifiers__(name='Gaussian_NB')
        bnb = Ensemble.__Classifiers__(name='Bernoulli_NB')
        dt = Ensemble.__Classifiers__(name='DecisionTree')
        bag = Ensemble.__Classifiers__(name='Bagging')
        Forest = Ensemble.__Classifiers__(name='RandomForest')
        Ada = Ensemble.__Classifiers__(name='AdaBoost')
        Gdbt = Ensemble.__Classifiers__(name='GradientBoosting')
        HGdbt = Ensemble.__Classifiers__(name='HistGradientBoosting')
        MLP = Ensemble.__Classifiers__(name='MLP')
        # Voting Classifier initialization
        vc = VotingClassifier(estimators=[('Gaussian_Process', GPC), ('Gaussian_NB', gnb),
                                          ('Bernoulli_NB', bnb), ('DecisionTree', dt), ('Bagging', bag),
                                          ('RandomForest', Forest), ('AdaBoost', Ada), ('GradientBoosting', Gdbt),
                                          ('HistGradientBoosting', HGdbt), ('MLP', MLP)
                                          ], voting='soft')

        # Fitting the vc model
        vc.fit(self.x_train, self.y_train)

        # Getting train and test accuracies from meta_model
        y_pred_train = vc.predict(self.x_train)
        y_pred = vc.predict(self.x_test)
        print(y_pred_train)
        print(f"Train accuracy: {accuracy_score(self.y_train, y_pred_train)}")
        print(f"Test accuracy: {accuracy_score(self.y_test, y_pred)}")
        print(f"评价指标:{classification_report(self.y_test, y_pred)}")
        l = []
        n = []
        label1 = []
        for clf, label in zip([GPC, gnb, bnb, dt, bag, Forest, Ada, Gdbt, HGdbt, MLP],
                              ['Neighbors', 'Gaussian_Process', 'Gaussian_NB', 'Bernoulli_NB', 'DecisionTree',
                               'Bagging', 'RandomForest', 'AdaBoost', 'GradientBoosting', 'HistGradientBoosting',
                               'MLP']):
            # cross_val_score中cv是交叉验证生成器或可迭代的次数
            scores = cross_val_score(clf, self.x_train, y_pred_train, cv=5, scoring='accuracy')
            l.append(scores.mean())
            n.append(scores.std())
            label1.append(label)
            # print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
        data = {'label': label1, '准确率': l, '偏移值': n, }
        frame = pd.DataFrame(data)
        frame.to_csv(fnameresult1, sep=' ', index=0, header=0)

        print(frame)


if __name__ == "__main__":
    # features = ['AC', 'ACC', 'DP', 'DR', 'KMER', 'PC-PseAAC', 'PC-PseAAC-General', 'PDT', 'SC-PseAAC',
    # 'SC-PseAAC-General']

    text = 'D:/桌面/科研/数据预处理/Test.txt'
    train = 'D:/桌面/科研/数据预处理/train.txt'
    fnameresult = '高尔基体结果.txt'
    ensemble = Ensemble()
    ensemble.load_data(text, train)
    ensemble.__VotingClassifier__(fnameresult)
