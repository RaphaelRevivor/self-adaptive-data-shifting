import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.font_manager
from sklearn import svm

class OCSVM(object):

    def fit(self, data, nu, gamma):
        print ("Model fitting...")
        clf = svm.OneClassSVM(nu = nu, kernel="rbf", gamma = gamma) # This setting can work at manually thicked data
        clf.fit(data)
        return clf

    def testset_generation(self, data):
        # select specific columns from the loaded table
        print ("Testset_generating...")

        # Generate uniform distributed novel observations for tesing uses
        len_X_test = len(data)//10
        min_x = min(data[:, 0])
        max_x = max(data[:, 0])
        min_y = min(data[:, 1])
        max_y = max(data[:, 1])

        X_test = np.zeros([len_X_test,2])
        X_test = [elem.tolist() for elem in X_test]
        X_test = np.array(X_test)
        margin_x = max_x - min_x
        margin_y = max_y - min_y
        X_test[:, 0] = np.random.uniform(low=min_x - 0.1*margin_x, high=max_x + 0.1*margin_x, size=len_X_test)
        X_test[:, 1] = np.random.uniform(low=min_y - 0.1*margin_y, high=max_y + 0.1*margin_y, size=len_X_test)

        return X_test

    def predict(self, clf, testset):
        print ("Model predicting...")
        y_pred_test = clf.predict(testset)

        return y_pred_test

    def visualization(self, clf, trainset, predicted_testset, target_size):
        print ("Calculating decision boundaries...")
        x, y = trainset[:, 0], trainset[:, 1]

        min_x = min(trainset[:, 0])
        max_x = max(trainset[:, 0])
        min_y = min(trainset[:, 1])
        max_y = max(trainset[:, 1])

        h = 0.02
        x_min, x_max = min_x - 1, max_x + 1
        y_min, y_max = min_y - 1, max_y + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        print ("Plotting graphs...")
        plt.figure(figsize=(12,9))
        # plt.title("Novelty Detection Using One Class SVM")

        legend_box = []
        legend_title = []

        # plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.PuBu)
        a1 = plt.contour(xx, yy, Z, levels=[0], linewidths=2, linestyles='solid', colors='red')
        a2 = plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors='orange')
        # b1 = plt.scatter(trainset[:, 0], trainset[:, 1], c='k', s=20)
        legend_box.append(a1.collections[0])
        # legend_box.append(b1)
        legend_title.append("Decision boundary")
        # legend_title.append("Training data")

        if len(predicted_testset):
            X_train= predicted_testset[:target_size,:]
            X_outlier = predicted_testset[target_size:,:]

            b2 = plt.scatter(X_outlier[:, 0], X_outlier[:, 1], c='b', s=20, lw=0)
            legend_box.append(b2)
            legend_title.append("Pseudo outlier")
            b3 = plt.scatter(X_train[:, 0], X_train[:, 1], c='g', s=20, lw=0)
            legend_box.append(b3)
            legend_title.append("Pseudo target data")            

        plt.legend(legend_box,
                   legend_title,
                   loc="upper left",
                   prop=matplotlib.font_manager.FontProperties(size=15))

        plt.savefig('one_class_svm.png')
        plt.show()

        return 0