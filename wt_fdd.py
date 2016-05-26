import numpy as np
import datetime as dt
import pandas as pd
import numpy.lib.recfunctions as rec
from sklearn import preprocessing as prep
from sklearn import cross_validation as cval
from sklearn import utils


class WT_data(object):
    def __init__(
            self, scada_data_file='Source Data/SCADA_data.csv',
            status_data_wec_file='Source Data/status_data_wec.csv',
            status_data_rtu_file='Source Data/status_data_rtu.csv',
            warning_data_wec_file='Source Data/warning_data_wec.csv',
            warning_data_rtu_file='Source Data/warning_data_rtu.csv'):
        """Initialises the class instance.

        Imports the data and returns arrays of SCADA & status data by
        calling import_data().

        Returns an array of fault-free SCADA data by calling
        fault_free_scada_data().

        Parameters
        ----------
        scada_data_file: str, optional
            The raw SCADA data csv file.
        status_data_wec_file: str, optional
            The status/fault csv file for the WEC
        status_data_rtu_file: str, optional
            The status/fault csv file for the RTU
        warning_data_wec_file: str, optional
            The warning/information csv file for the WEC
        warning_data_rtu_file: str, optional
            The warning/information csv file for the RTU

        Returns
        -------
        scada_data: ndarray
            The imported and correctly formatted SCADA data
        status_data_wec: ndarray
            The imported and correctly formatted WEC status data
        status_data_rtu: ndarray
            The imported and correctly formatted RTU status data
        warning_data_wec: ndarray
            The imported and correctly formatted WEC warning data
        warning_data_rtu: ndarray
            The imported and correctly formatted RTU warning data
        fault_free_scada_data: ndarray
            The fault free scada data. Filtered according to the
            following rules:

            - selects only data according to certain "good" wec statuses
            (shown below). Note it selects only data 1800s after a
            change to a "good" wec status, and 7200s before a change to
            any other wec status
            - selects only data according to a single "good" rtu status.
            Note it selects only data 600s after a change to the "good"
            rtu status, and 600s before a change to any other rtu status
            - Finally, it filters out the 10h warning data. This is an
            rtu_warning with a 'Main_Warning' of 230. It's an ambiguous
            warning which I estimate to mean that there may or may not
            be curtailment (for a variety of reasons) in the 10 hours
            following when this status comes into play.

        Notes
        -----
        Both status_data_wec.csv & status_data_rtu.csv originally come from
        pes_extrainfo.csv, filtered according to their plant number.
        SCADA_data.csv contains the wsd, 03d and 04d data files all
        combined together.
        """
        self.scada_data_file = scada_data_file
        self.status_data_wec_file = status_data_wec_file
        self.status_data_rtu_file = status_data_rtu_file
        self.warning_data_wec_file = warning_data_wec_file
        self.warning_data_rtu_file = warning_data_rtu_file

        # Import the data using the default folder structure above
        self.__import_data()

        # Filter out and extract the fault-free data from the imported
        # SCADA data
        self.__get_fault_free_scada_data()

    def __import_data(self):
        """This imports the data, and returns arrays of SCADA, status &
        warning data. Dates are converted to unix time, and strings are
        encoded in the correct format (unicode). Two new fields,
        "Inverter_averages" and "Inverter_std_dev", are also added to
        the SCADA data. These are the average and standard deviation of
        all Inverter Temperature fields.

        Returns
        -------
        scada_data: ndarray
            The imported and correctly formatted SCADA data
        status_data_wec: ndarray
            The imported and correctly formatted WEC status data
        status_data_rtu: ndarray
            The imported and correctly formatted RTU status data
        warning_data_wec: ndarray
            The imported and correctly formatted WEC warning data
        warning_data_rtu: ndarray
            The imported and correctly formatted RTU warning data
        """
        scada_data = np.genfromtxt(
            open(self.scada_data_file, 'rb'), dtype=(
                '<U19', '<f4', '<f4', '<f4', '<f4', '<f4', '<f4', '<f4', '<f4',
                '<f4', '<f4', '<f4', '<f4', '<f4', '<f4', '<f4', '<f4', '<f4',
                '<f4', '<f4', '<f4', '<f4', '<f4', '<f4', '<f4', '<f4', '<f4',
                '<f4', '<f4', '<f4', '<f4', '<f4', '<f4', '<f4', '<f4', '<f4',
                '<f4', '<f4', '<f4', '<f4', '<f4', '<f4', '<f4', '<f4', '<f4',
                '<f4', '<f4', '<f4', '<f4', '<f4', '<f4', '<f4', '<f4', '<f4',
                '<f4', '<f4', '<f4', '<f4', '<f4', '<f4', '<f4', '<f4', '<f4'),
            delimiter=",", names=True)

        status_data_wec = np.genfromtxt(
            open(self.status_data_wec_file, 'rb'), dtype=(
                '<U19', '<i4', '<i4', '<U9', '<U63', '<i4', '|b1', '|b1',
                '<f4'), delimiter=",", names=True)

        status_data_rtu = np.genfromtxt(
            open(self.status_data_rtu_file, 'rb'), dtype=(
                '<U19', '<i4', '<i4', '<U9', '<U63', '<i4', '|b1', '|b1',
                '<f4'), delimiter=",", names=True)

        warning_data_wec = np.genfromtxt(
            open(self.warning_data_wec_file, 'rb'), dtype=(
                '<U19', '<i4', '<i4', '<U9', '<U63', '|b1', '<f4'),
            delimiter=",", names=True)

        warning_data_rtu = np.genfromtxt(
            open(self.warning_data_rtu_file, 'rb'), dtype=(
                '<U19', '<i4', '<i4', '<U9', '<U63', '|b1', '<f4'),
            delimiter=",", names=True)

        data_files = (scada_data, status_data_rtu, status_data_wec,
                      warning_data_rtu, warning_data_wec)
        for data_file in data_files:
            # Convert datetimes to Unix timestamps (as strings)
            time = data_file['Time']
            for i in range(0, len(time)):
                t = dt.datetime.strptime(time[i], "%d/%m/%Y %H:%M:%S")
                t = (t - dt.datetime.fromtimestamp(3600)).total_seconds()
                time[i] = t

        # convert Unix timestamp string to float (for some reason this
        # doesn't work when in the loop above):
        dtlist = scada_data.dtype.descr
        dtlist[0] = (dtlist[0][0], '<f4')
        dtlist = np.dtype(dtlist)
        scada_data = scada_data.astype(dtlist)

        dtlist = status_data_wec.dtype.descr
        dtlist[0] = (dtlist[0][0], '<f4')
        dtlist = np.dtype(dtlist)
        self.status_data_wec = status_data_wec.astype(dtlist)

        dtlist = status_data_rtu.dtype.descr
        dtlist[0] = (dtlist[0][0], '<f4')
        dtlist = np.dtype(dtlist)
        self.status_data_rtu = status_data_rtu.astype(dtlist)

        dtlist = warning_data_wec.dtype.descr
        dtlist[0] = (dtlist[0][0], '<f4')
        dtlist = np.dtype(dtlist)
        self.warning_data_wec = warning_data_wec.astype(dtlist)

        dtlist = warning_data_rtu.dtype.descr
        dtlist[0] = (dtlist[0][0], '<f4')
        dtlist = np.dtype(dtlist)
        self.warning_data_rtu = warning_data_rtu.astype(dtlist)

        # Add 2 extra columns to scada - Inverter_averages and
        # Inverter_std_dev, as features
        inverters = np.array([
            'CS101__Sys_1_inverter_1_cabinet_temp',
            'CS101__Sys_1_inverter_2_cabinet_temp',
            'CS101__Sys_1_inverter_3_cabinet_temp',
            'CS101__Sys_1_inverter_4_cabinet_temp',
            'CS101__Sys_1_inverter_5_cabinet_temp',
            'CS101__Sys_1_inverter_6_cabinet_temp',
            'CS101__Sys_1_inverter_7_cabinet_temp',
            'CS101__Sys_2_inverter_1_cabinet_temp',
            'CS101__Sys_2_inverter_2_cabinet_temp',
            'CS101__Sys_2_inverter_3_cabinet_temp',
            'CS101__Sys_2_inverter_4_cabinet_temp'])
        means = pd.DataFrame(scada_data[inverters]).mean(axis=1).values
        stds = pd.DataFrame(scada_data[inverters]).std(axis=1).values
        self.scada_data = rec.append_fields(scada_data, [
            'Inverter_averages', 'Inverter_std_dev'],
            data=[means, stds], usemask=False)

    def filter(
            self, scada_data, sw_data, sw_column_name,
            filter_type='fault_free', return_inverse=False,
            time_delta_1=3600, time_delta_2=7200, *sw_codes):
        """Returns SCADA data which correspond to certain times around
        when certain statuses or warnings came into effect on the
        turbine.

        This function takes the scada data obtained in import_data,
        or a subset of that data, and filters it corresponding to
        certain status or warning codes in the status or warning data.

        Depending on filter_type, indices returned may refer to times of
        fault-free or faulty operation, or times leading up to faulty
        operation. See the filter_type parameter description for
        details. See also __fault_free_filter(),
        __fault_case_1_filter(), __fault_case_2_filter() and
        __fault_case_3_filter() for details on how each specific case is
        handled.

        Parameters
        ----------
        scada_data: ndarray
            SCADA data to be filtered. Must be the scada data obtained
            from import_data(), or a subset of that data.
        sw_data: ndarray
            Status/Warning data to be used for reference timestamps.
            One of: warning_data_rtu, warning_data_wec, status_data_rtu,
            status_data_wec.
        sw_column_name: string
            Refers to the column being filtered, i.e. 'Main_Status',
            'Full_Status', 'Main_Warning' or 'Full_Warning'.
        filter_type: string, optional (default='fault_free')
            - If 'fault_free', the function will return indices of
            scada_data which fall between time_delta_1 after the start
            of each passed sw_code when it appears in the sw_data, and
            time_delta_2 before it ends.
            - If 'fault_case_1', the function gets a timestamp of
            time_delta_1 before the start of a certain status/warning
            which corresponds to faulty operation, and time_delta_2
            after the status/warning ends. It returns indices of
            scada_data which fall between these time stamps.
            - If 'fault_case_2', the function gets timestamps for the
            times between time_delta_1 and time_delta_2 before the start
            of a certain status/warning which corresponds to the start
            of faulty operation. It returns indices of scada_data which
            fall between these time stamps.
            - If 'fault_case_3' The function gets timestamps for the
            times between time_delta_1 and time_delta_2 before a certain
            fault starts. It returns indices of scada_data which fall
            between these time stamps, but ONLY IF no other instance of
            the same fault occured during this period. Therefore, it
            contains only data which led up to the fault. Used for fault
            prediction purposes.
        return_inverse: boolean, optional (default=False)
            If True, the function will return the indices of filtered
            SCADA data which DON'T correspond to what this function
            would normally return. I.e. the function would select all
            the indices of SCADA data as normal, and then instead return
            all OTHER indices. E.g. the function could find indices of
            when the turbine was in fault-free operation, and then
            return all OTHER indices instead.
        time_delta_1: integer, optional (default=3600)
            - If filter_type = 'fault_free', it's the time AFTER normal
            operation begins from which to include scada_data indices
            - If filter_type = 'fault_case_1', this is the amount of
            time before the fault from which to include the returned
            scada fault data.
            - If filter_type = 'fault_case_2' or 'fault_case_3', this
            refers to the time BEFORE faulty operation begins from which
            to include scada_data indices. Must be greater than or equal
            to time_delta_2.
        time_delta_2: integer, optional (default=7200)
            - If filter_type = 'fault_free', it's the time BEFORE normal
            operation ENDS from which to include scada_data indices
            - If filter_type = 'fault_case_1', AFTER faulty operation
            ends from which to include scada_data indices
            - If filter_type = 'fault_case_2' or 'fault_case_3', this
            refers to the time AFTER faulty operation begins from which
            to stop including scada_data indices. Must be less than
            time_delta_1
        *sw_codes: array of str or int
            The set of codes to be filtered. Can be a single value or an
            array. If sw_column_name is 'Full_Status' or 'Full_Warning',
            then it must be a set of strings referring to the
            statuses/warnings (e.g. full status '0 : 0' for nominal
            operation). If sw_column_name is 'Main_Status' or
            'Main_Warning', then it must be a set of integers referring
            to the statuses/warnings (e.g. main status 62 for feeding
            faults)

        Returns
        -------
        filtered_scada_data: ndarray
            If filter_type = False, filtered_scada_data should be data
            strictly corresponding to fault-free data.
            If filter_type = True, filtered_scada_data is data strictly
            corresponding to fault data.
        """
        # Aggregate all the indices of sw_data from the passed sw_codes
        # together:
        sw_data_indices = np.array([], dtype='i')
        for sw_code in sw_codes:
            sw = np.where((
                sw_data[sw_column_name] == sw_code))
            sw_data_indices = np.sort(
                np.concatenate((sw_data_indices, sw[0])))

        if filter_type == 'fault_free':
            filtered_scada_indices = self.__fault_free_filter(
                scada_data, sw_data, sw_data_indices, time_delta_1,
                time_delta_2)
        elif filter_type == 'fault_case_1':
            filtered_scada_indices = self.__fault_case_1_filter(
                scada_data, sw_data, sw_data_indices, time_delta_1,
                time_delta_2)
        elif filter_type == 'fault_case_2':
            filtered_scada_indices = self.__fault_case_2_filter(
                scada_data, sw_data, sw_data_indices, time_delta_1,
                time_delta_2)
        elif filter_type == 'fault_case_3':
            filtered_scada_indices = self.__fault_case_3_filter(
                scada_data, sw_data, sw_data_indices, time_delta_1,
                time_delta_2)
        else:
            raise ValueError(
                'filter_type must be one of \'fault_free\', '
                '\'fault_case_1\', \'fault_case_2\' or \'fault_case_3\'.')

        if return_inverse is False:
            return scada_data[filtered_scada_indices]
        elif return_inverse is True:
            # using a mask is the simplest way I know to get the inverse
            mask = np.array([True]).repeat(len(scada_data))
            mask[filtered_scada_indices] = False
            return scada_data[mask]
        else:
            raise ValueError('return_inverse must be True or False')

    def __fault_free_filter(
            self, scada_data, sw_data, sw_data_indices, time_delta_1,
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
            SCADA data to be filtered. Must be the scada data obtained
            from import_data(), or a subset of that data.
        sw_data: ndarray
            Status/Warning data to be used for reference timestamps. One
            of: warning_data_rtu, warning_data_wec, status_data_rtu,
            status_data_wec.
        sw_data_indices: ndarray
            Indices of the sw_data whose timestamps will be used
            to match up with scada_data. Should be obtained from
            WT_data.filter() function.
        time_delta_1: integer
            Time AFTER normal operation begins from which to include
            scada_data indices
        time_delta_2: integer
            Time BEFORE normal operation ENDS from which to include
            scada_data indices
        Returns
        -------
        fault_free_scada_indices: ndarray
            indices of scada_data which correspond to fault-free
            operation
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
            # -1], then we use time_delta_2 before scada_data['Time'][
            # -1] as the upper time limit for finding
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

    def __fault_case_1_filter(
            self, scada_data, sw_data, sw_data_indices, time_delta_1,
            time_delta_2):
        """Returns indices of SCADA data corresponding to faulty
        operation under a certain fault, according to case_1 in the
        `fault_filtering <fault-filtering>`_ documentation.

        The function gets a timestamp of time_delta_1 before the start
        of a certain status/warning which corresponds to faulty operation,
        and time_delta_2 after the status/warning ends. It returns
        indices of scada_data which fall between these time stamps.

        Parameters
        ----------
        scada_data: ndarray
            SCADA data to be filtered. Must be the scada data obtained
            from import_data(), or a subset of that data.
        sw_data: ndarray
            Status/Warning data to be used for reference timestamps. One
            of: warning_data_rtu, warning_data_wec, status_data_rtu,
            status_data_wec.
        sw_data_indices: ndarray
            Indices of the sw_data whose timestamps will be used
            to match up with scada_data. Should be obtained from
            WT_data.filter() function.
        time_delta_1: integer
            Time BEFORE faulty operation begins from which to include
            scada_data indices
        time_delta_2: integer
            Time AFTER faulty operation ends from which to include
            scada_data indices
        Returns
        -------
        fault_scada_indices: ndarray
            indices of scada_data which correspond to fault-free
            operation
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

    def __fault_case_2_filter(
            self, scada_data, sw_data, sw_data_indices, time_delta_1,
            time_delta_2):
        """Returns indices of SCADA data leading up to a certain fault,
        according to case_2 in the `fault_filtering <fault-filtering>`_
        documentation.

        The function gets timestamps for the times between time_delta_1
        and time_delta_2 before the start of a certain status/warning
        which corresponds to the start of faulty operation. It returns
        indices of scada_data which fall between these time stamps.

        Parameters
        ----------
        scada_data: ndarray
            SCADA data to be filtered. Must be the scada data obtained
            from import_data(), or a subset of that data.
        sw_data: ndarray
            Status/Warning data to be used for reference timestamps. One
            of: warning_data_rtu, warning_data_wec, status_data_rtu,
            status_data_wec.
        sw_data_indices: ndarray
            Indices of the sw_data whose timestamps will be used
            to match up with scada_data. Should be obtained from
            WT_data.filter() function.
        time_delta_1: integer
            Time BEFORE faulty operation begins from which to include
            scada_data indices. Must be greater than or equal to
            time_delta_2
        time_delta_2: integer
            Time before faulty operation begins from which to stop
            including scada_data indices. Must be less than time_delta_1
        Returns
        -------
        fault_scada_indices: ndarray
            indices of scada_data which correspond to fault-free
            operation
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

    def __fault_case_3_filter(
            self, scada_data, sw_data, sw_data_indices, time_delta_1,
            time_delta_2):
        """Returns indices of SCADA data leading up to a certain fault,
        according to case_3 in the `fault_filtering <fault-filtering>`_
        documentation.

        The function gets timestamps for the times between time_delta_1
        and time_delta_2 before a certain fault starts. It returns
        indices of scada_data which fall between these time stamps,
        but ONLY IF no other instance of this fault occured during this
        period. Therefore, it contains only data of normal operation
        (or possibly under faulty operation, but of a different fault)
        which led up to the fault.

        Parameters
        ----------
        scada_data: ndarray
            SCADA data to be filtered. Must be the scada data obtained
            from import_data(), or a subset of that data.
        sw_data: ndarray
            Status/Warning data to be used for reference timestamps. One
            of: warning_data_rtu, warning_data_wec, status_data_rtu,
            status_data_wec.
        sw_data_indices: ndarray
            Indices of the sw_data whose timestamps will be used
            to match up with scada_data. Should be obtained from
            WT_data.filter() function.
        time_delta_1: integer
            Time BEFORE faulty operation begins from which to include
            scada_data indices. Must be greater than or equal to
            time_delta_2
        time_delta_2: integer
            Time before faulty operation begins from which to stop
            including scada_data indices. Must be less than time_delta_1
        Returns
        -------
        fault_scada_indices: ndarray
            indices of scada_data which correspond to fault-free
            operation
        """
        if time_delta_1 < time_delta_2:
            raise ValueError("time_delta_1 must be greater than or equal to "
                             "time_delta_2!")
        fault_scada_indices = np.array([], dtype='i')
        # filtered_scada_instances for fault data are only returned
        # between time_delta_1 and time_delta_2 before a fault, if the
        # same type of fault does not occur in that period.

        # the first fault instance (or, if there's only one fault
        # instance) will no overlap with a previous one, so selecting
        # these indices is straighforward:
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

    def __get_fault_free_scada_data(self):
        """Uses WT_data.filter() to fault free data, according to
        certain criteria.

        Returns
        -------
        fault_free_scada_data: ndarray
            The fault free scada data. Filtered according to the
            following rules:

            - selects only data according to certain "good" wec statuses
            (shown below). Note it selects only data 1800s after a
            change to a "good" wec status, and 7200s before a change to
            any other wec status
            - selects only data according to a single "good" rtu status.
            Note it selects only data 600s after a change to the "good"
            rtu status, and 600s before a change to any other rtu status
            - Finally, it filters out the 10h warning data. This is an
            rtu_warning with a 'Main_Warning' of 230. It's an ambiguous
            warning which I estimate to mean that there may or may not
            be curtailment (for a variety of reasons) in the 10 hours
            following when this status comes into play.
        """

        # These are the statuses that correspond to nominal wec operation:
        statuses = ('0 : 0', '2 : 1', '2 : 2', '3 : 12')

        # Filtering to only include the above statuses.
        scada_good_wec = self.filter(
            self.scada_data, self.status_data_wec, 'Full_Status', 'fault_free',
            False, 1800, 7200, *statuses)

        # Further filtering to only include good rtu statuses
        scada_good_status = self.filter(
            scada_good_wec, self.status_data_rtu, 'Full_Status', 'fault_free',
            False, 600, 600, '0 : 0')

        #Final filtering to not include the 10h warning (see docs for details)
        scada_good_status_10h = self.filter(
            scada_good_status, self.warning_data_wec, 'Main_Warning',
            'fault_case_1', True, 600, 36700, 230)

        self.fault_free_scada_data = scada_good_status_10h

        # print(
        #     "Note that the fault free data is selected according to the"
        #     " following criteria:")
        # print(
        #     "- selects only data according to certain \"good\" wec statuses"
        #     "(see docs for details). Note it selects only data 1800s after a "
        #     "change to a \"good\" wec status, and 7200s before a change to any"
        #     " other wec status")
        # print(
        #     "- selects only data according to a single \"good\" rtu status."
        #     "Note it selects only data 600s after a change to the \"good\" rtu"
        #     "status, and 600s before a change to any other rtu status")
        # print(
        #     "- Finally, it filters out the 10h warning data (more on this in"
        #     "the docs)")

    def get_all_fault_data(self, filter_type='fault_case_1',
                           time_delta_1=600, time_delta_2=600):
        """This function is a shortcut to get a list of faults using the
        WT_data.filter() function. The faults that are included are the
        most frequently occuring faults in the enercon Ringaskiddy data
        up to end March 2015.

        Parameters
        ----------
        filter_type: string, optional (default='fault_case_1')
            The filter_type to be passed to the WT_data.filter()
            function. Must be one of 'fault_case_1', 'fault_case_2' or
            'fault_case_3'. See the WT_data.filter() function for
            details
        time_delta_1: integer
            The time_delta_1 to be passed to the WT_data.filter()
            function. If filter_type = 'fault_case_2' or 'fault_case_3',
            time_delta_1 must be greater than or equal to time_delta_2.
            See the WT_data.filter() function for details
        time_delta_2: integer
            The time_delta_2 to be passed to the WT_data.filter()
            function. If filter_type = 'fault_case_2' or 'fault_case_3',
            time_delta_2 must be less than time_delta_1. See the
            WT_data.filter() function for details

        Returns
        -------
        all_faults_scada_data: ndarray
            An array of all SCADA data corresponding to fault times
        feeding_fault_scada_data: ndarray
            An array of SCADA data corresponding to feeding faults
        aircooling_fault_scada_data: ndarray
            An array of SCADA data corresponding to aircooling faults
        excitation_fault_scada_data: ndarray
            An array of SCADA data corresponding to excitation faults
        generator_heating_fault_scada_data: ndarray
            An array of SCADA data corresponding to generator heating
            faults
        mains_failure_fault_scada_data: ndarray
            An array of SCADA data corresponding to mains_failure faults
        """

        # Main status of the faults to be included:
        faults = (80, 62, 228, 60, 9)

        if filter_type not in ('fault_case_1', 'fault_case_2', 'fault_case_3'):
            raise ValueError('filter_type must be one of \'fault_case_1\', '
                             '\'fault_case_2\' or \'fault_case_3\'.')

        all_faults_scada_data = self.filter(
            self.scada_data, self.status_data_wec, "Main_Status", filter_type,
            False, time_delta_1, time_delta_2, *faults)
        feeding_fault_scada_data = self.filter(
            self.scada_data, self.status_data_wec, "Main_Status", filter_type,
            False, time_delta_1, time_delta_2, 62)
        mains_failure_fault_scada_data = self.filter(
            self.scada_data, self.status_data_wec, "Main_Status", filter_type,
            False, time_delta_1, time_delta_2, 60)
        aircooling_fault_scada_data = self.filter(
            self.scada_data, self.status_data_wec, "Main_Status", filter_type,
            False, time_delta_1, time_delta_2, 228)
        excitation_fault_scada_data = self.filter(
            self.scada_data, self.status_data_wec, "Main_Status", filter_type,
            False, time_delta_1, time_delta_2, 80)
        generator_heating_fault_scada_data = self.filter(
            self.scada_data, self.status_data_wec, "Main_Status", filter_type,
            False, time_delta_1, time_delta_2, 9)

        return all_faults_scada_data, feeding_fault_scada_data, \
            mains_failure_fault_scada_data, aircooling_fault_scada_data, \
            excitation_fault_scada_data, generator_heating_fault_scada_data

    @staticmethod
    def get_test_train_data(fault_free_scada_data_set, features,
                            fault_data_sets, normalize=True, split=0.2):
        """Generate labels for the SCADA data.
        Parameters
        ----------
        fault_free_scada_data_set: ndarray (default=
                                    Enercon.fault_free_scada_data)
            Array of fault-free data obtained using the WT_data.filter()
            function. Should use WT_data.fault_free_scada_data.
            See WT_data.__get_fault_free_scada_data() for details.
        fault_data_sets: list of ndarrays
            list of  arrays of subsets of fault data obtained using the
            WT_data.filter() function.
            Example 1:
            >>> fault_data_sets = [feeding_fault_scada_data,
                                   aircooling_fault_scada_data,
                                   excitation_fault_scada_data]
            Example 2:
            >>> fault_data_sets = [feeding_fault_scada_data]
        normalize: Boolean, optional (default=True)
            Whether or not to normalize the training data.
        split: float, optional (default=0.2)
            The ratio of testing : training data to use.

        Returns
        -------
        X_train: ndarray
            Training data samples
        y_train: ndarray
            Training data labels
        X_test: ndarray
            Testing data samples
        y_test: ndarray
            Testing data labels
        X_train_bal: ndarray
            Balanced training data samples (i.e. no. of fault-free
            samples = sum of no. of each fault class samples)
        y_train_bal: ndarray
            Balanced training data labels (i.e. no. of fault-free
            samples = sum of no. of each fault class samples)
        """
        if type(fault_data_sets) is not list:
            raise TypeError(
                "fault_data_sets must be a list of arrays of fault data. "
                "Examples:\n"
                "Example 1:\n"
                ">>> fault_data_sets = [feeding_fault_scada_data,\n"
                "                       aircooling_fault_scada_data,\n"
                "                       excitation_fault_scada_data]\n\n"
                "Example 2:\n"
                ">>> fault_data_sets = [feeding_fault_scada_data]")

        fault_free_labels = np.zeros(len(fault_free_scada_data_set), dtype=int)

        # we create the final_data_set array which will eventually
        # contain both fault and fault-free data, anbd their labels. We
        # need them with their labels in the same array because when we
        # shuffle it all up we want the labels to remain correct
        final_data_set = rec.append_fields(
            fault_free_scada_data_set[features], ['label'],
            data=[fault_free_labels], usemask=False)
        # append the "fault" data set(s) to the final_data_set
        i = 1
        for fault_data_set in fault_data_sets:
            labels = np.array([i]).repeat(len(fault_data_set))
            fault_data_set = rec.append_fields(
                fault_data_set[features], ['label'], data=[labels],
                usemask=False)
            final_data_set = np.concatenate([final_data_set, fault_data_set])
            i += 1
        # shuffle it all up to make it totez random lolz
        np.random.shuffle(final_data_set)

        # here we're just separating out the labels from the data for,
        # cos that's what sklearn wants from us.
        y = final_data_set['label']
        # drop that extra field since it's no longer needed
        X = rec.drop_fields(final_data_set, ['label'], False).view(
            np.float32).reshape(len(final_data_set),
                                len(final_data_set.dtype) - 1)
        # finally, we get to create that training and test data!
        if normalize is True:
            X_norm = prep.normalize(X)
            X_train, X_test, y_train, y_test = cval.train_test_split(
                X_norm, y, test_size=split)
        else:
            X_train, X_test, y_train, y_test = cval.train_test_split(
                X, y, test_size=split)

        # shuffle again for the balanced training data, i.e. when no. fault
        # examples = no. fault-free examples, in case we want to
        # compare against the unbalanced performance. URRDAY I'M SHUFFLIN'
        X_train_bal, y_train_bal = utils.shuffle(X_train, y_train)

        # Create the balanced training sets
        X_train_bad_bal = X_train_bal[np.where(y_train_bal != 0)]
        y_train_bad_bal = y_train_bal[np.where(y_train_bal != 0)]
        X_train_good_bal = X_train_bal[
            np.where(y_train_bal == 0)][0:len(X_train_bad_bal)]
        y_train_good_bal = y_train_bal[
            np.where(y_train_bal == 0)][0:len(X_train_bad_bal)]

        X_train_bal = np.concatenate([X_train_good_bal, X_train_bad_bal])
        y_train_bal = np.concatenate([y_train_good_bal, y_train_bad_bal])

        X_train_bal, y_train_bal = utils.shuffle(X_train_bal, y_train_bal)

        return X_train, X_test, y_train, y_test, X_train_bal, y_train_bal
