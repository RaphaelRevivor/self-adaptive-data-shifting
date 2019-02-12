import numpy as np
import csv
import copy
import timeit

from ocsvm import OCSVM
from edge_pattern_detection import Data_Shifting

filename = 'banana.csv'
with open(filename, 'r') as f:
    reader = csv.reader(f)
    data = list(reader)

data = np.array(data)
data = data[1:].astype(np.float)

new_data = data[data[:,2] == 2]

new_data = new_data[:, [0,1]]
# only reserve unique elements here
new_data = np.unique(new_data, axis=0)

# generate artificial outlier and target datasets
data_shifting = Data_Shifting(new_data)
pseudo_outliers = data_shifting.outlier_generation()
pseudo_targets = data_shifting.target_generation()

# initialization for error calculations
nu_list = [0.01, 0.02, 0.05, 0.08, 0.1]
gamma_list = [2,5,8,10,15]

ocsvm = OCSVM()
error_array = np.zeros((len(nu_list),len(gamma_list)))
full_err_array = np.zeros((len(nu_list),len(gamma_list), 2))
grid_size = len(nu_list)*len(gamma_list)
err_min = 1.0
best_err = [0.0, 0.0]
best_param = [0.0,0.0]

training_time_sum = 0.0
predicting_time_outliers = 0.0
predicting_time_targets = 0.0

# grid search
for index, i in enumerate(nu_list):
    for jndex, j in enumerate(gamma_list):
        print("nu=%r, gamma=%r"%(i,j))

        # model fitting
        start = timeit.default_timer()
        clf = ocsvm.fit(new_data, i, j)
        stop = timeit.default_timer()
        training_time_sum += stop - start

        # predicting
        start = timeit.default_timer()
        y_outliers = ocsvm.predict(clf, pseudo_outliers)
        stop = timeit.default_timer()
        predicting_time_outliers += stop - start

        start = timeit.default_timer()
        y_targets = ocsvm.predict(clf, pseudo_targets)
        stop = timeit.default_timer()
        predicting_time_targets += stop - start

        # calculate the error
        err_outliers = float(y_outliers[y_outliers[:]==1].shape[0]) / y_outliers.shape[0]
        err_targets = float(y_targets[y_targets[:]!=1].shape[0]) / y_targets.shape[0]
        err = 0.5*err_outliers + 0.5*err_targets
        # save results
        error_array[index][jndex] = err
        full_err_array[index][jndex] = [err_outliers, err_targets]
        print("err_outliers=%r, err_targets=%r, err=%r"%(err_outliers,err_targets,err))
        if err < err_min:
            err_min = err
            best_param = [i,j]
            best_err = [err_outliers, err_targets]

best_param = np.array(best_param)
print ("The error array is: %r"%error_array)
print ("The full error array is: %r"%full_err_array)
print ("The smallest error is: %r"%err_min)
print ("The corresponding outlier/target err are: %r"%best_err)
print ("The best param combination is: %r"%best_param)
print ("Training set size: ")
print(new_data.shape)
print ("Pseudo target set size: ")
print(pseudo_targets.shape)
print ("Pseudo outlier set size: ")
print (pseudo_outliers.shape)

# plot the best model
start = timeit.default_timer()
clf = ocsvm.fit(new_data,best_param[0], best_param[1])
stop = timeit.default_timer()
final_train = stop - start

# # predicting
start = timeit.default_timer()
y_targets = ocsvm.predict(clf, pseudo_targets)
stop = timeit.default_timer()
final_target = stop - start

start = timeit.default_timer()
y_outliers = ocsvm.predict(clf, pseudo_outliers)
stop = timeit.default_timer()
final_outlier = stop - start

targets = np.column_stack((pseudo_targets, y_targets))
outliers = np.column_stack((pseudo_outliers, y_outliers))
pred_testset = np.concatenate((targets, outliers))
ocsvm.visualization(clf, new_data, pred_testset, pseudo_targets.shape[0])

writer = csv.writer(open("result_ocsvm.csv", 'w'))
writer.writerow(['Best param:'])
writer.writerow(best_param)
writer.writerow(['Smallest error:'])
writer.writerow([err_min])
writer.writerow(['Best error:'])
writer.writerow(best_err)
writer.writerow(['Error array:'])
for row in error_array:    
  writer.writerow(row)
writer.writerow(['Full error array:'])
for row in full_err_array:    
  writer.writerow(row)
writer.writerow(['Average fitting time:'])
writer.writerow([training_time_sum/grid_size])
writer.writerow(['Average predicting time for outliers:'])
writer.writerow([predicting_time_outliers/grid_size])
writer.writerow(['Average predicting time for targets:'])
writer.writerow([predicting_time_targets/grid_size])
writer.writerow(['Best model training/predict_target/predict_outlier:'])
writer.writerow([final_train,final_target, final_outlier])
