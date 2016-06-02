# # Set the Feature Set
# F=['WEC_ava_windspeed',
# 'WEC_ava_Rotation',
# 'WEC_ava_Power',
# #'WEC_ava_Nacel_position_including_cable_twisting',
# 'WEC_ava_reactive_Power',
# 'WEC_ava_blade_angle_A',
# 'Inverter_averages',
# 'Inverter_std_dev',
# 'CS101__Spinner_temp',
# 'CS101__Front_bearing_temp',
# 'CS101__Rear_bearing_temp',
# 'CS101__Pitch_cabinet_blade_A_temp',
# 'CS101__Pitch_cabinet_blade_B_temp',
# 'CS101__Pitch_cabinet_blade_C_temp',
# 'CS101__Rotor_temp_1',
# 'CS101__Rotor_temp_2',
# 'CS101__Stator_temp_1',
# 'CS101__Stator_temp_2',
# 'CS101__Nacelle_ambient_temp_1',
# 'CS101__Nacelle_ambient_temp_2',
# 'CS101__Nacelle_temp',
# 'CS101__Nacelle_cabinet_temp',
# 'CS101__Main_carrier_temp',
# 'CS101__Rectifier_cabinet_temp',
# 'CS101__Yaw_inverter_cabinet_temp',
# 'CS101__Fan_inverter_cabinet_temp',
# 'CS101__Ambient_temp',
# 'CS101__Tower_temp',
# 'CS101__Control_cabinet_temp',
# 'CS101__Transformer_temp'
#     ]

import winfault

Enercon = winfault.WT_data()

# -------------The following tests the filtering function:---------------------
statuses = ('0 : 0', '2 : 1', '2 : 2', '3 : 12')

scada_good_wec = Enercon.filter(
    Enercon.scada_data, Enercon.status_data_wec, 'Full_Status', 'fault_free',
    False, 1800, 7200, *statuses)

# Should be 39713:
print("Scada_good_wec, should be 39713: ", len(scada_good_wec))

scada_good_status = Enercon.filter(
    scada_good_wec, Enercon.status_data_rtu, 'Full_Status', 'fault_free',
    False, 600, 600, '0 : 0')

# Should be 36387:
# Note, the value obtained for the equivalent function in the "Import
# and Label turbine data" jupyter notebook for the following is 29095.
# It's smaller because that function doesn't include data up to the end
# of the time period (see the else: statement in
# WT_data.__fault_free_filter() for details)
print("scada_good_status, should be 36387: ", len(scada_good_status))

# Should be 32056:
# Note, the value obtained for the equivalent function in the "Import
# and Label turbine data" jupyter notebook for the following is 28682.
# It's smaller because the previous function in the notebook doesn't
# include data up to the end of the time period (see the else: statement
# in WT_data.__fault_free_filter() for details). Hence, in this case we
# would be filtering down from a smaller amount of data than we have
# here.
scada_good_status_10h = Enercon.filter(
    scada_good_status, Enercon.warning_data_wec, 'Main_Warning',
    'fault_case_1', True, 600, 36700, 230)
print("scada_good_status_10h, should be 32056: ", len(scada_good_status_10h))

# Should be 32056:
print('Enercon.fault_free_scada_data should be 32056: ',
      len(Enercon.fault_free_scada_data))

print('\n \n')

# ------------------------Testing the fault filtering--------------------------
scada_data = Enercon.scada_data
filter_type = "fault_case_1"
sw_data = Enercon.status_data_wec
sw_column_name = "Main_Status"
time_delta_1 = 600
time_delta_2 = 600
faults = (80, 62, 228, 60, 9)
all_faults_scada_data = Enercon.filter(
    scada_data, sw_data, sw_column_name, filter_type, False, time_delta_1,
    time_delta_2, *faults)
feeding_fault_scada_data = Enercon.filter(
    scada_data, sw_data, sw_column_name, filter_type, False, time_delta_1,
    time_delta_2, 62)
mains_failure_fault_scada_data = Enercon.filter(
    scada_data, sw_data, sw_column_name, filter_type, False, time_delta_1,
    time_delta_2, 60)
aircooling_fault_scada_data = Enercon.filter(
    scada_data, sw_data, sw_column_name, filter_type, False, time_delta_1,
    time_delta_2, 228)
excitation_fault_scada_data = Enercon.filter(
    scada_data, sw_data, sw_column_name, filter_type, False, time_delta_1,
    time_delta_2, 80)
generator_heating_fault_scada_data = Enercon.filter(
    scada_data, sw_data, sw_column_name, filter_type, False, time_delta_1,
    time_delta_2, 9)

print("all_faults_scada_data, should be 454: ", len(all_faults_scada_data))
print("feeding_fault_scada_data, should be 263: ",
      len(feeding_fault_scada_data))
print("mains_failure_fault_scada_data, should be 20: ",
      len(mains_failure_fault_scada_data))
print("aircooling_fault_scada_data, should be  62: ",
      len(aircooling_fault_scada_data))
print("excitation_fault_scada_data, should be 178: ",
      len(excitation_fault_scada_data))
print("generator_heating_fault_scada_data, should be 44: ",
      len(generator_heating_fault_scada_data))

print('\n \n')

# ----------------------Testing the Fault Data---------------------------------
all_faults_scada_data, feeding_fault_scada_data, \
    mains_failure_fault_scada_data, aircooling_fault_scada_data, \
    excitation_fault_scada_data, generator_heating_fault_scada_data = \
    Enercon.get_all_fault_data()

print("all_faults_scada_data, should be 454: ", len(all_faults_scada_data))
print("feeding_fault_scada_data, should be 263: ",
      len(feeding_fault_scada_data))
print("mains_failure_fault_scada_data, should be 20: ",
      len(mains_failure_fault_scada_data))
print("aircooling_fault_scada_data, should be  62: ",
      len(aircooling_fault_scada_data))
print("excitation_fault_scada_data, should be 178: ",
      len(excitation_fault_scada_data))
print("generator_heating_fault_scada_data, should be 44: ",
      len(generator_heating_fault_scada_data))

print('\n \n')
# ----------------------Testing Exception Handling-----------------------------
try:
    # This should raise a ValueError for wrong filter_type
    Enercon.filter(
        scada_data, sw_data, "Main_Status", 'fault_case_4', False,
        600, 600, 60)
except ValueError as e:
    print('Should raise ValueError about filter_type \n', e)
    print('\n \n')
try:
    # This should raise a ValueError for wrong return_inverse
    Enercon.filter(
        scada_data, sw_data, "Main_Status", 'fault_case_1', 'Flalse',
        600, 600, 60)
except ValueError as e:
    print('This should raise a ValueError for wrong return_inverse \n', e)
    print('\n \n')
try:
    # Should raise ValueError fault_case_2 time_delta1 > time_delta_2
    Enercon.filter(
        scada_data, sw_data, "Main_Status", 'fault_case_2', False,
        400, 1200, 60)
except ValueError as e:
    print('Should raise ValueError fault_case_2 time_delta1 > time_delta_2 \n',
          e)
    print('\n \n')
try:
    # Should raise a ValueError for fault_case_3 time_delta_1 > time_delta_2
    Enercon.filter(
        scada_data, sw_data, "Main_Status", 'fault_case_3', False,
        400, 1200, 60)
except ValueError as e:
    print('Should raise ValueError fault_case_3 time_delta1 > time_delta_2 \n',
          e)
    print('\n \n')
try:
    # Should raise ValueError filter_type has to be fault_case_1, 2, 3:
    Enercon.get_all_fault_data('flault blah')
except ValueError as e:
    print('Should raise ValueError filter_type has to be fault_case_1, 2, 3',
          '\n', e)
    print('\n \n')
try:
    # Should raise ValueError about filter_type, time_delta_1 > time_delta_2
    Enercon.get_all_fault_data('fault_case_2', 300)
except ValueError as e:
    print('Should raise ValueError fault_case_2/3 time_delta1 > time_delta_2',
          '\n', e)
