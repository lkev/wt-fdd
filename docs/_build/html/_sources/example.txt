Usage Example
=============

Import using ``import winfault``.

Create the class, and access correctly formatted SCADA data::

    Turbine = winfault.WT_data()
    scada_data = Turbine.scada_data
    fault_free_data = Turbine.fault_free_scada_data

Access columns by typing ``data['column name']``. The following gets the first 100 windspeeds of the fault_free_data::

	fault_free_data['WEC_ava_windspeed'][0:99]

Get all the fault data using default values (type help(Turbine.filter) or help(Turbine.get_all_fault_data) for details)::

	all_faults_scada_data, feeding_fault_scada_data, \
    mains_failure_fault_scada_data, aircooling_fault_scada_data, \
    excitation_fault_scada_data, generator_heating_fault_scada_data = \
    Turbine.get_all_fault_data()

Generate training and testing data for fault-free, feeding fault and mains failure faults, using specific features::

	features = ['Time',
            'WEC_ava_windspeed',
            'WEC_ava_Rotation',
            'WEC_ava_Power',
            'WEC_ava_reactive_Power',
            'WEC_ava_blade_angle_A',
            'Inverter_averages',
            'Inverter_std_dev']

	fault_data_sets = [feeding_fault_scada_data, mains_failure_fault_scada_data]
	X_train, X_test, y_train, y_test, X_train_bal, y_train_bal = \
    Turbine.get_test_train_data(features, fault_data_sets, fault_free_data,
                                True, 0.2)
