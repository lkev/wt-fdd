import winfault
import warnings
import numpy as np
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV

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
X_train, X_test, y_train, y_test, X_train_bal, y_train_bal = \
    Turbine.get_test_train_data(features, faults, nf)

# labels for confusion matrix
labels = ['no-fault', 'feeding fault', 'excitation fault', 'generator fault']

print("========================================================")
print("------Building models using balanced training data------")
print("========================================================")

# set the parameter space (class_weight is None for the balanced training data)
parameter_space_bal = {
    'kernel': ['linear', 'rbf', 'poly'], 'gamma': ['auto', 1e-3, 1e-4],
    'C': [0.01, .1, 1, 10, 100, 1000], 'class_weight': [None]}

# train and test svm
clf_bal, bgg_bal = winfault.svm_class_and_score(
    X_train_bal, y_train_bal, X_test, y_test, labels,
    parameter_space=parameter_space_bal, bagged=True, score='recall_weighted',
    search_type=GridSearchCV)

print("==========================================================")
print("------Building models using imbalanced training data------")
print("==========================================================")
# set the parameter space (class_weight is None for the balanced training data)
parameter_space = {
    'kernel': ['linear', 'rbf', 'poly'], 'gamma': ['auto', 1e-3, 1e-4],
    'C': [0.01, .1, 1, 10, 100, 1000],
    'class_weight': [
        {0: 0.01}, {1: 1}, {1: 2}, {1: 10}, {1: 50}, 'balanced']}

# train and test svm
clf, bgg = winfault.svm_class_and_score(
    X_train, y_train, X_test, y_test, labels,
    parameter_space=parameter_space, bagged=True, score='recall_weighted',
    search_type=RandomizedSearchCV)

print("============================================================")
print("----------Training for detection of general faults----------")
print("============================================================")
print("============================================================", "\n\n")

# need to change this to the original way it was done!!!

# af = np.append(ff, ef)
# af = np.append(af, gf)

# X_train, X_test, y_train, y_test, X_train_bal, y_train_bal = \
#     Turbine.get_test_train_data(features, [af], nf)

# # labels for confusion matrix
# labels = ['no-fault', 'fault']

# print("========================================================")
# print("------Building models using balanced training data------")
# print("========================================================")

# set the parameter space (class_weight is None for the balanced training data)
# parameter_space_bal = {
#     'kernel': ['linear', 'rbf', 'poly'], 'gamma': ['auto', 1e-3, 1e-4],
#     'C': [0.01, .1, 1, 10, 100, 1000], 'class_weight': [None]}

# train and test svm
# clf_bal, bgg_bal = winfault.svm_class_and_score(
#     X_train_bal, y_train_bal, X_test, y_test, labels,
#     parameter_space=parameter_space_bal, bagged=True, score='recall_weighted',
#     search_type=GridSearchCV)

# print("==========================================================")
# print("------Building models using imbalanced training data------")
# print("==========================================================")
# set the parameter space (class_weight is None for the balanced training data)
# parameter_space = {
#     'kernel': ['linear', 'rbf', 'poly'], 'gamma': ['auto', 1e-3, 1e-4],
#     'C': [0.01, .1, 1, 10, 100, 1000],
#     'class_weight': [
#         {0: 0.01}, {1: 1}, {1: 2}, {1: 10}, {1: 50}, 'balanced']}

# # train and test svm
# clf, bgg = winfault.svm_class_and_score(
#     X_train, y_train, X_test, y_test, labels,
#     parameter_space=parameter_space, bagged=True, score='recall_weighted',
#     search_type=RandomizedSearchCV)
