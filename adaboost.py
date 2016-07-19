import winfault
import warnings
import numpy as np
import sklearn
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC

%matplotlib inline

Turbine = winfault.WT_data()

scada = Turbine.scada_data

# warnings suppressed because there's loads of UndefinedMetricWarnings
warnings.filterwarnings("ignore")

features = ['WEC_ava_windspeed',
            'WEC_ava_Rotation',
            'WEC_ava_Power',
            'WEC_ava_reactive_Power',
            'WEC_ava_blade_angle_A',
            'Inverter_averages',
            'Inverter_std_dev',
            'CS101__Spinner_temp',
            'CS101__Front_bearing_temp',
            'CS101__Rear_bearing_temp',
            'CS101__Pitch_cabinet_blade_A_temp',
            'CS101__Pitch_cabinet_blade_B_temp',
            'CS101__Pitch_cabinet_blade_C_temp',
            'CS101__Rotor_temp_1',
            'CS101__Rotor_temp_2',
            'CS101__Stator_temp_1',
            'CS101__Stator_temp_2',
            'CS101__Nacelle_ambient_temp_1',
            'CS101__Nacelle_ambient_temp_2',
            'CS101__Nacelle_temp',
            'CS101__Nacelle_cabinet_temp',
            'CS101__Main_carrier_temp',
            'CS101__Rectifier_cabinet_temp',
            'CS101__Yaw_inverter_cabinet_temp',
            'CS101__Fan_inverter_cabinet_temp',
            'CS101__Ambient_temp',
            'CS101__Tower_temp',
            'CS101__Control_cabinet_temp',
            'CS101__Transformer_temp']

# This gets all the data EXCEPT the faults listed. Labels as nf for "no-fault"
nf = Turbine.filter(scada, Turbine.status_data_wec, "Main_Status",
                    'fault_case_1', True, 600, 600, [62, 9, 80])
# feeding fault
ff = Turbine.filter(scada, Turbine.status_data_wec, "Main_Status",
                    'fault_case_1', False, 600, 600, 62)
# generator heating fault
gf = Turbine.filter(scada, Turbine.status_data_wec, "Main_Status",
                    'fault_case_1', False, 600, 600, 9)
# excitation fault
ef = Turbine.filter(scada, Turbine.status_data_wec, "Main_Status",
                    'fault_case_1', False, 600, 600, 80)

print("=============================================================")
print("----------Training for detection of specific faults----------")
print("=============================================================")
print("=============================================================", "\n")

# select the faults to include.
faults = [ff, ef, gf]

# label and split into train, test and balanced training data
xtrain, xtest, ytrain, ytest, xbaltrain, ybaltrain = \
    Turbine.get_test_train_data(features, faults, nf)

# labels for confusion matrix
labels = ['no-fault', 'feeding fault', 'excitation fault', 'generator fault']

print("========================================================")
print("------Building models using balanced training data------")
print("========================================================")

# train and test the SVM

parameter_space_bal = {
    'kernel': ['linear', 'rbf', 'poly'], 'gamma': ['auto', 1e-3, 1e-4],
    'C': [0.01, .1, 1, 10, 100, 1000], 'class_weight': [None]}

print("Building balanced SVM")
SVM_bal = RandomizedSearchCV(SVC(C=1), parameter_space_bal, cv=10,
        scoring='recall_weighted', iid=True)
print("fitting balanced SVM")
SVM_bal.fit(xbaltrain, ybaltrain)

print("Hyperparameters for balanced SVM found:")
print(SVM_bal.best_params_)

print("getting predictions for balanced SVM")
y_pred_svm_bal = SVM_bal.predict(xtest)

print("\n\n results for SVM")
winfault.clf_scoring(ytest, y_pred_svm_bal, labels)

print("========================================================")
print("------Building models using Imbalanced training data------")
print("========================================================")
parameter_space = {
    'kernel': ['linear', 'rbf', 'poly'], 'gamma': ['auto', 1e-3, 1e-4],
    'C': [0.01, .1, 1, 10, 100, 1000],
    'class_weight': [
        {0: 0.01}, {1: 1}, {1: 2}, {1: 10}, {1: 50}, 'balanced']}

print("Building Imbalanced SVM")
SVM = RandomizedSearchCV(SVC(C=1), parameter_space, cv=10,
                         scoring='recall_weighted', iid=True)
print("fitting Imbalanced SVM")
SVM.fit(xtrain, ytrain)

print("Hyperparameters for Imbalanced SVM found:")
print(SVM.best_params_)

print("getting predictions for Imbalanced SVM")
y_pred_svm = SVM.predict(xtest)

print("\n\n results for SVM")
winfault.clf_scoring(ytest, y_pred_svm, labels)

# train and test adaboost svm

print("Building AdaBoost Classifier")
adaboost = sklearn.ensemble.AdaBoostClassifier(
    base_estimator=SVC(**SVM.best_params_), algorithm='SAMME')

print("fitting AdaBoost Classifier")
adaboost.fit(xbaltrain, ybaltrain)

print("getting predictions")
y_pred_ada = adaboost.predict(xtest)

print("\n\nResults for AdaBoosted SVM:")
winfault.clf_scoring(ytest, y_pred_ada, labels)

# train and test svm
# clf_bal, bgg_bal = winfault.svm_class_and_score(
#     xbaltrain, ybaltrain, xtest, ytest, labels,
#    parameter_space=parameter_space_bal, bagged=True, score='recall_weighted',
#     search_type=GridSearchCV)
