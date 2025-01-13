from sklearn.datasets import load_iris
from sklearn.model_selection import TimeSeriesSplit, ShuffleSplit, StratifiedKFold, LeaveOneOut, cross_val_score, KFold, LeavePOut

iris=load_iris()
X=iris.data
Y=iris.target
# 1.Kfold
kf=KFold(n_splits=5)
# 2.Stratified K-fold
skf = StratifiedKFold(n_splits=5)
# 3.LeavePOut 每次分割2个作为训练集，剩下的作为验证集，
# 这个过程重复进行，直到整个数据集被划分为p-样本的验证集和n-p训练样本。
op = LeavePOut(p=2)
op.get_n_splits(X)

# 4.LeaveOneOut，每次1个作为train剩下的作为test
lo = LeaveOneOut()

# 5.Monte Carlo Cross-Validation
# Shuffle Cross-Validation 30%作为train，50%作为test，重复10次分割
ss = ShuffleSplit(train_size=0.3, test_size=0.5, n_splits=10)

# 6.从一个小的数据子集开始，作为训练集。基于这个数据集，预测以后的数据点，
# 然后检查准确性，预测的样本被作为下一个训练数据集的一部分，并对后续的样本进行预测
ts = TimeSeriesSplit()
print(ts)
print("---")
for train_index, test_index in ts.split(X):
    print(train_index)
    print(test_index)
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]
# 注意，这里的logreg是验证的模型对象
# score=cross_val_score(logreg,X,Y,cv=kf)