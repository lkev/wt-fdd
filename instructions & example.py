import wt_fdd

# Create the class
# See help(Enercon) for details on what it returns.
Enercon = wt_fdd.WT_data()

# Correctly formatted scada_data:
# Note 'Time' column is in unix time.
scada_data = Enercon.scada_data
# Column names:
for d in scada_data.dtype.descr:
    print(d)

# fault-free data set:
# Note it has same column names (i.e. all of them)
fault_free_data = Enercon.fault_free_scada_data

# Access columns by typing data['column_name']
# The following gets the first 100 windspeeds of the fault_free_data:
fault_free_data['WEC_ava_windspeed'][0:99]

# Get all the fault data:
# Type help(Enercon.filter) or help(Enercon.get_all_fault_data) for
# details. Default values are used here.
all_faults_scada_data, feeding_fault_scada_data, \
    mains_failure_fault_scada_data, aircooling_fault_scada_data, \
    excitation_fault_scada_data, generator_heating_fault_scada_data = \
    Enercon.get_all_fault_data()

# Want to look at the following features (for example):
features = ['Time',
            'WEC_ava_windspeed',
            'WEC_ava_Rotation',
            'WEC_ava_Power',
            'WEC_ava_reactive_Power',
            'WEC_ava_blade_angle_A',
            'Inverter_averages',
            'Inverter_std_dev']

# Want to generate training and testing data for fault-free, feeding
# fault and mains failure faults:
fault_data_sets = [feeding_fault_scada_data, mains_failure_fault_scada_data]

# Convert to test/train data:
# Note the _bal suffix means balanced (i.e. no. fault-free samples = sum
# of no. of each fault class)
X_train, X_test, y_train, y_test, X_train_bal, y_train_bal = \
    Enercon.get_test_train_data(fault_free_data, features,
                                fault_data_sets, True, 0.2)
