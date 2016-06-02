import winfault

Enercon = winfault.WT_data()

all_faults_scada_data, feeding_fault_scada_data, \
    mains_failure_fault_scada_data, aircooling_fault_scada_data, \
    excitation_fault_scada_data, generator_heating_fault_scada_data = \
    Enercon.get_all_fault_data()

features = ['WEC_ava_windspeed',
            'WEC_ava_Rotation',
            'WEC_ava_Power',
            # 'WEC_ava_Nacel_position_including_cable_twisting',
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

# fault_data_sets = [feeding_fault_scada_data,
#                    aircooling_fault_scada_data,
#                    excitation_fault_scada_data]
fault_data_sets = [feeding_fault_scada_data]

X_train, X_test, y_train, y_test, X_train_bal, y_train_bal = \
    Enercon.get_test_train_data(features, fault_data_sets)

a = len(X_train) + len(X_test)
b = len(Enercon.fault_free_scada_data)
for fault_data_set in fault_data_sets:
    b = len(fault_data_set) + b

print(a == b)
