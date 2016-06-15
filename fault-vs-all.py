import winfault

Turbine = winfault.WT_data()

scada = Turbine.scada_data

# no-fault
nf = Turbine.filter(scada,Turbine.status_data_wec, "Main_Status",
                     'fault_case_1', True, 600,600,[62,60,80,9,228])
# feeding fault
ff = Turbine.filter(scada,Turbine.status_data_wec, "Main_Status",
                    'fault_case_1', False, 600,600,62)
# mains failure fault
mf = Turbine.filter(scada,Turbine.status_data_wec, "Main_Status",
                    'fault_case_1', False, 600,600,60)
# aircooling fault
af = Turbine.filter(scada,Turbine.status_data_wec, "Main_Status",
                    'fault_case_1', False, 600,600,80)
# generator heating fault
gf = Turbine.filter(scada,Turbine.status_data_wec, "Main_Status",
                    'fault_case_1', False, 600,600,9)
# excitation fault
ef = Turbine.filter(scada,Turbine.status_data_wec, "Main_Status",
                    'fault_case_1', False, 600,600,228)

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
# select the faults to include
faults = [ff, mf]
# label and split into train, test and balanced training data
xtrain, xtest, ytrain, ytest, xbaltrain, ybaltrain = Turbine.get_test_train_data(features, faults, nf)
# labels for confusion matrix
labels = ['no-fault', 'feeding fault', 'mains fault']
# train and test svm
winfault.svm_class_and_score(xbaltrain, ybaltrain, xtest, ytest, labels)
