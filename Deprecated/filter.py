import numpy as np


def __filter(
        scada_data, sw_data, sw_column_name, filter_type='fault_free',
        time_delta_1=3600, time_delta_2=7200, *sw_codes):
    """Returns indices of SCADA data which correspond to certain times
    around when certain statuses or warnings came into effect on the
    turbine.

    This function takes the SCADA data obtained in import_data,
    or a subset of that data, and filters it corresponding to certain
    status or warning codes (given by sw_codes) in the status or warning
    data (given by sw_data), e.g. status_rtu, warning_wec, etc.

    Depending on filter_type, indices returned may refer to times of
    fault-free or faulty operation, or times leading up to faulty
    operation. See `fault_filtering <fault-filtering>`_ for details. See
    also __fault_free_filter(), __fault_case_1_filter(),
    __fault_case_2_filter() and __fault_case_3_filter() for details on
    how each specific case is handled.

    Parameters
    ----------
    scada_data: ndarray
        SCADA data to be filtered. Must be the scada data obtained from
        import_data(), or a subset of that data.
    sw_data: ndarray
        Status/Warning data to be used for reference timestamps. One of:
        warning_rtu, warning_wec, status_rtu, status_wec.
    sw_column_name: string
        Refers to the column being filtered, i.e. 'Main_Status',
        'Full_Status', 'Main_Warning' or 'Full_Warning'.
    filter_type: string, optional (default=fault_free')
        Refers to whether the passed sw_codes refer to fault
        data or fault-free data. If 'fault_free', then the function
        will assume the sw_codes provided correspond to nominal
        operation. If 'fault_case_1', 'fault_case_2' or 'fault_case_3',
        it's assumed sw_codes refer to faults. See
        `fault_filtering <fault-filtering>`_ for details
    time_delta_1: integer, optional
        If filter_type = 'fault_free', it's the time AFTER normal
        operation begins from which to include scada_data indices
        If filter_type = 'fault_case_1', this is the amount of time
        before the fault from which to include the returned scada fault
        data.
        If filter_type = 'fault_case_2' or 'fault_case_3', this refers
        to the time BEFORE faulty operation begins from which to include
        scada_data indices. Must be greater than or equal to
        time_delta_2.
    time_delta_2: integer, optional
        If filter_type = 'fault_free', it's the time BEFORE normal
        operation ENDS from which to include scada_data indices
        If filter_type = 'fault_case_1', AFTER faulty operation ends
        from which to include scada_data indices
        If filter_type = 'fault_case_2' or 'fault_case_3', this refers
        to the time AFTER faulty operation begins from which to stop
        including scada_data indices. Must be less than time_delta_1
    *sw_codes: array of str or int
        The set of codes to be filtered. Can be a single value or an
        array. If sw_column_name is 'Full_Status' or 'Full_Warning,
        then it must be a set of strings referring to the
        statuses/warnings (e.g. full status '0 : 0' for nominal
        operation). If sw_column_name is 'Main_Status' or
        'Main_Warning', then it must be a set of integers referring to
        the statuses/warnings (e.g. main status 62 for feeding faults)

    Returns
    -------
    filtered_scada_data: ndarray
        If filter_type = False, filtered_scada_data should be data
        strictly corresponding to fault-free data.
        If filter_type = True, filtered_scada_data is data strictly
        corresponding to fault data.
    """
    # Get the indices of sw_data which correspond to the passed sw_codes:
    sw_data_indices = np.array([], dtype='i')
    for sw_code in sw_codes:
        sw = np.where((
            sw_data[sw_column_name] == sw_code))
        sw_data_indices = np.sort(
            np.concatenate((sw_data_indices, sw[0])))
    print('sw data length: ', len(sw_data_indices))

    if filter_type == 'fault_free':
        filtered_scada_indices = __fault_free_filter(
            scada_data, sw_data, sw_data_indices, time_delta_1, time_delta_2)
    elif filter_type == 'fault_case_1':
        filtered_scada_indices = __fault_case_1_filter(
            scada_data, sw_data, sw_data_indices, time_delta_1, time_delta_2)
    elif filter_type == 'fault_case_2':
        # filtered_scada_indices for fault data are between time_delta_1
        # and time_delta_2 before each instance of sw_data_indices:
        filtered_scada_indices = __fault_case_2_filter(
            scada_data, sw_data, sw_data_indices, time_delta_1, time_delta_2)
    elif filter_type == 'fault_case_3':
        # filtered_scada_instances for fault data are only returned
        # between time_delta_1 and time_delta_2 before a fault, if the
        # same type of fault does not occur in that period.

        # the first fault instance (or, if there's only one fault
        # instance) is treated differently:
        filtered_scada_indices = __fault_case_3_filter(
            scada_data, sw_data, sw_data_indices, time_delta_1, time_delta_2)
    else:
        raise ValueError(
            'filter_type must be one of \'fault_free\', '
            '\'fault_case_1\', \'fault_case_2\' or \'fault_case_3\'.')

    return scada_data[filtered_scada_indices]


def __fault_free_filter(scada_data, sw_data, sw_data_indices, time_delta_1,
                        time_delta_2):
    """Returns indices of SCADA data corresponding to fault-free
    operation.

    The function gets a timestamp of time_delta_1 after the start of
    each status/warning which correspond to fault-free operation, and
    time_delta_2 before the status/warning ends. It returns indices of
    scada_data which fall between these time stamps.

    Parameters
    ----------
    scada_data: ndarray
        SCADA data to be filtered. Must be the scada data obtained from
        import_data(), or a subset of that data.
    sw_data: ndarray
        Status/Warning data to be used for reference timestamps. One of:
        warning_rtu, warning_wec, status_rtu, status_wec.
    sw_data_indices: ndarray
        Indices of the sw_data whose timestamps will be used
        to match up with scada_data. Should be obtained from __filter()
        function.
    time_delta_1: integer
        Time AFTER normal operation begins from which to include
        scada_data indices
    time_delta_2: integer
        Time BEFORE normal operation ENDS from which to include
        scada_data indices
    Returns
    -------
    fault_free_scada_indices: ndarray
        indices of scada_data which correspond to fault-free operation
    """
    fault_free_scada_indices = np.array([], dtype='i')
    for sw_data_index in sw_data_indices:
        # fault_free_scada_indices for fault-free data are normally
        # between time_delta_1 AFTER each instance of
        # sw_data_indices, and time_delta_2 BEFORE the next general
        # entry of sw_data (i.e. sw_data_indices + 1):
        if sw_data[sw_data_index] != sw_data[-1]:
            sf = np.where(
                (scada_data['Time'] >=
                    sw_data['Time'][sw_data_index] + time_delta_1) &
                (scada_data['Time'] <
                    sw_data['Time'][sw_data_index + 1] - time_delta_2))
        # However, if the current sw_data_index represents sw_data[
        # -1], then we use time_delta_2 before scada_data['Time'][-1]
        # as the upper time limit for finding
        # fault_free_scada_indices. This is because sw_data[
        # sw_data_index + 1] does not exist, and we don't know if
        # the sw_code will change after scada_data['Time'][-1]:
        else:
            sf = np.where(
                (scada_data['Time'] >=
                    sw_data['Time'][sw_data_index] + time_delta_1) &
                (scada_data['Time'] <
                    scada_data['Time'][-1] - time_delta_2))

        fault_free_scada_indices = np.concatenate(
            (fault_free_scada_indices, sf[0]), axis=0)

    fault_free_scada_indices = np.unique(fault_free_scada_indices)

    return fault_free_scada_indices


def __fault_case_1_filter(scada_data, sw_data, sw_data_indices, time_delta_1,
                          time_delta_2):
    """Returns indices of SCADA data corresponding to faulty operation
    under a certain fault, according to case_1 in the
    `fault_filtering <fault-filtering>`_ documentation.

    The function gets a timestamp of time_delta_1 before the start of
    a certain status/warning which corresponds to faulty operation, and
    time_delta_2 after the status/warning ends. It returns indices of
    scada_data which fall between these time stamps.

    Parameters
    ----------
    scada_data: ndarray
        SCADA data to be filtered. Must be the scada data obtained from
        import_data(), or a subset of that data.
    sw_data: ndarray
        Status/Warning data to be used for reference timestamps. One of:
        warning_rtu, warning_wec, status_rtu, status_wec.
    sw_data_indices: ndarray
        Indices of the sw_data whose timestamps will be used
        to match up with scada_data. Should be obtained from __filter()
        function.
    time_delta_1: integer
        Time BEFORE faulty operation begins from which to include
        scada_data indices
    time_delta_2: integer
        Time AFTER faulty operation ends from which to include
        scada_data indices
    Returns
    -------
    fault_scada_indices: ndarray
        indices of scada_data which correspond to fault-free operation
    """
    fault_scada_indices = np.array([], dtype='i')

    for sw_data_index in sw_data_indices:
        # fault_scada_indices for fault data are normally
        # between time_delta_1 BEFORE each instance of
        # sw_data_indices, and time_delta_2 AFTER the next general
        # entry of sw_data (i.e. sw_data_indices + 1):
        if sw_data[sw_data_index] != sw_data[-1]:
            sf = np.where(
                (scada_data['Time'] >=
                    sw_data['Time'][sw_data_index] - time_delta_1) &
                (scada_data['Time'] <
                    sw_data['Time'][sw_data_index + 1] + time_delta_2))
        # However, if the current sw_data_index represents sw_data[
        # -1], then we use time_delta_2 before scada_data['Time'][-1]
        # as the upper time limit for finding
        # fault_scada_indices. This is because sw_data[
        # sw_data_index + 1] does not exist, and we don't know if
        # the sw_code will change after scada_data['Time'][-1]:
        else:
            sf = np.where(
                (scada_data['Time'] >=
                    sw_data['Time'][sw_data_index] - time_delta_1) &
                (scada_data['Time'] <
                    scada_data['Time'][-1]))

        fault_scada_indices = np.concatenate(
            (fault_scada_indices, sf[0]), axis=0)

    fault_scada_indices = np.unique(fault_scada_indices)

    return fault_scada_indices


def __fault_case_2_filter(scada_data, sw_data, sw_data_indices, time_delta_1,
                          time_delta_2):
    """Returns indices of SCADA data leading up to a certain fault,
    according to case_2 in the `fault_filtering <fault-filtering>`_
    documentation.

    The function gets timestamps for the times between time_delta_1 and
    time_delta_2 before the start of a certain status/warning which
    corresponds to the start of faulty operation. It returns indices of
    scada_data which fall between these time stamps.

    Parameters
    ----------
    scada_data: ndarray
        SCADA data to be filtered. Must be the scada data obtained from
        import_data(), or a subset of that data.
    sw_data: ndarray
        Status/Warning data to be used for reference timestamps. One of:
        warning_rtu, warning_wec, status_rtu, status_wec.
    sw_data_indices: ndarray
        Indices of the sw_data whose timestamps will be used
        to match up with scada_data. Should be obtained from __filter()
        function.
    time_delta_1: integer
        Time BEFORE faulty operation begins from which to include
        scada_data indices. Must be greater than or equal to
        time_delta_2
    time_delta_2: integer
        Time AFTER faulty operation begins from which to stop including
        scada_data indices. Must be less than time_delta_1
    Returns
    -------
    fault_scada_indices: ndarray
        indices of scada_data which correspond to fault-free operation
    """
    if time_delta_1 < time_delta_2:
        raise ValueError("time_delta_1 must be greater than or equal to "
                         "time_delta_2!")
    fault_scada_indices = np.array([], dtype='i')
    # fault_scada_indices for fault data are between time_delta_1
    # and time_delta_2 before each instance of sw_data_indices:
    for sw_data_index in sw_data_indices:
        sf = np.where(
            (scada_data['Time'] >=
                sw_data['Time'][sw_data_index] - time_delta_1) &
            (scada_data['Time'] <
                sw_data['Time'][sw_data_index] - time_delta_2))

        fault_scada_indices = np.concatenate(
            (fault_scada_indices, sf[0]), axis=0)

    fault_scada_indices = np.unique(fault_scada_indices)

    return fault_scada_indices


def __fault_case_3_filter(scada_data, sw_data, sw_data_indices, time_delta_1,
                          time_delta_2):
    """Returns indices of SCADA data leading up to a certain fault,
    according to case_3 in the `fault_filtering <fault-filtering>`_
    documentation.

    The function gets timestamps for the times between time_delta_1 and
    time_delta_2 before a certain fault starts. It returns indices of
    scada_data which fall between these time stamps, but ONLY IF no
    other instance of this fault occured during this period. Therefore,
    it contains only data of normal operation (or possibly under faulty
    operation, but of a different fault) which led up to the fault.

    Parameters
    ----------
    scada_data: ndarray
        SCADA data to be filtered. Must be the scada data obtained from
        import_data(), or a subset of that data.
    sw_data: ndarray
        Status/Warning data to be used for reference timestamps. One of:
        warning_rtu, warning_wec, status_rtu, status_wec.
    sw_data_indices: ndarray
        Indices of the sw_data whose timestamps will be used
        to match up with scada_data. Should be obtained from __filter()
        function.
    time_delta_1: integer
        Time BEFORE faulty operation begins from which to include
        scada_data indices. Must be greater than or equal to
        time_delta_2
    time_delta_2: integer
        Time AFTER faulty operation begins from which to stop including
        scada_data indices. Must be less than time_delta_1
    Returns
    -------
    fault_scada_indices: ndarray
        indices of scada_data which correspond to fault-free operation
    """
    if time_delta_1 < time_delta_2:
        raise ValueError("time_delta_1 must be greater than or equal to "
                         "time_delta_2!")
    fault_scada_indices = np.array([], dtype='i')
    # filtered_scada_instances for fault data are only returned
    # between time_delta_1 and time_delta_2 before a fault, if the
    # same type of fault does not occur in that period.

    # the first fault instance (or, if there's only one fault
    # instance) will no overlap with a previous one, so selecting these
    # indices is straighforward:
    fault_scada_indices = np.where(
        (scada_data['Time'] >= sw_data[sw_data_indices[0]]['Time'] -
            time_delta_1) &
        (scada_data['Time'] < sw_data[sw_data_indices[0]]['Time'] -
            time_delta_2))[0]
    # The rest of the indices must be picked from times when the
    # previous fault instance does not overlap with the current
    # fault instance - time_delta_1:
    for i in range(1, len(sw_data_indices)):
        if (sw_data[sw_data_indices[i]]['Time'] - time_delta_1 >=
                sw_data[sw_data_indices[i - 1] + 1]['Time']):
            sf = np.where(
                (scada_data['Time'] >=
                    sw_data['Time'][sw_data_indices[i]] - time_delta_1) &
                (scada_data['Time'] <
                    sw_data['Time'][sw_data_indices[i]] - time_delta_2))

            fault_scada_indices = np.concatenate(
                (fault_scada_indices, sf[0]))

    fault_scada_indices = np.unique(fault_scada_indices)

    return fault_scada_indices
