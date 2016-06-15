# import winfault

# Turbine = winfault.WT_data()

# all_faults_scada_data, feeding_fault_scada_data, \
#     mains_failure_fault_scada_data, aircooling_fault_scada_data, \
#     excitation_fault_scada_data, generator_heating_fault_scada_data = \
#     Turbine.get_all_fault_data()

# features = ['WEC_ava_windspeed',
#             'WEC_ava_Rotation',
#             'WEC_ava_Power',
#             # 'WEC_ava_Nacel_position_including_cable_twisting',
#             'WEC_ava_reactive_Power',
#             'WEC_ava_blade_angle_A',
#             'Inverter_averages',
#             'Inverter_std_dev',
#             'CS101__Spinner_temp',
#             'CS101__Front_bearing_temp',
#             'CS101__Rear_bearing_temp',
#             'CS101__Pitch_cabinet_blade_A_temp',
#             'CS101__Pitch_cabinet_blade_B_temp',
#             'CS101__Pitch_cabinet_blade_C_temp',
#             'CS101__Rotor_temp_1',
#             'CS101__Rotor_temp_2',
#             'CS101__Stator_temp_1',
#             'CS101__Stator_temp_2',
#             'CS101__Nacelle_ambient_temp_1',
#             'CS101__Nacelle_ambient_temp_2',
#             'CS101__Nacelle_temp',
#             'CS101__Nacelle_cabinet_temp',
#             'CS101__Main_carrier_temp',
#             'CS101__Rectifier_cabinet_temp',
#             'CS101__Yaw_inverter_cabinet_temp',
#             'CS101__Fan_inverter_cabinet_temp',
#             'CS101__Ambient_temp',
#             'CS101__Tower_temp',
#             'CS101__Control_cabinet_temp',
#             'CS101__Transformer_temp']

fault_data_sets = [mains_failure_fault_scada_data]

# Convert to test/train data:
# Note the _bal suffix means balanced (i.e. no. fault-free samples = sum
# of no. of each fault class)
X_train, X_test, y_train, y_test, X_train_bal, y_train_bal = \
    Turbine.get_test_train_data(features, fault_data_sets)

winfault.svm_class_and_score(
    X_train_bal, y_train_bal, X_test, y_test)