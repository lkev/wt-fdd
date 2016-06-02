import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import pandas as pd
import numpy.lib.recfunctions as rec

from scipy.interpolate import splev, splrep
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC, OneClassSVM
from sklearn import preprocessing as prep


def import_data(
    scada_data='Source Data/SCADA_data.csv',
        status_data_wec='Source Data/status_data_wec.csv',
        status_data_rtu='Source Data/status_data_rtu.csv',
        warning_data_wec='Source Data/warning_data_wec.csv',
        warning_data_rtu='Source Data/warning_data_rtu.csv'):
    """This imports the data, and returns arrays of SCADA & status data.
    Dates are converted to unix time, and strings are encoded in the
    correct format (unicode). Two new fields, "Inverter_averages" and
    "Inverter_std_dev", are also added to the SCADA data. These are the
    average and standard deviation of all Inverter Temperature fields.

    Parameters
    ----------

    scada_data: str, optional
        The raw SCADA data csv file.
    status_data_wec: str, optional
        The status/fault csv file for the WEC
    status_data_rtu: str, optional
        The status/fault csv file for the RTU
    warning_data_wec: str, optional
        The warning/information csv file for the WEC
    warning_data_rtu: str, optional
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

    Extra Notes
    -----------
    Both status_wec.csv & status_rtu.csv originally come from
    pes_extrainfo.csv, filtered according to their plant number.
    SCADA_data.csv contains the wsd, 03d and 04d data files all combined
    together.
    """
    SCADA = np.genfromtxt(open(scada_data, 'rb'), dtype=(
        '<U19', '<f4', '<f4', '<f4', '<f4', '<f4', '<f4', '<f4', '<f4',
        '<f4', '<f4', '<f4', '<f4', '<f4', '<f4', '<f4', '<f4', '<f4',
        '<f4', '<f4', '<f4', '<f4', '<f4', '<f4', '<f4', '<f4', '<f4',
        '<f4', '<f4', '<f4', '<f4', '<f4', '<f4', '<f4', '<f4', '<f4',
        '<f4', '<f4', '<f4', '<f4', '<f4', '<f4', '<f4', '<f4', '<f4',
        '<f4', '<f4', '<f4', '<f4', '<f4', '<f4', '<f4', '<f4', '<f4',
        '<f4', '<f4', '<f4', '<f4', '<f4', '<f4', '<f4', '<f4', '<f4'),
        delimiter=",", names=True)

    status_wec = np.genfromtxt(open(status_data_wec, 'rb'), dtype=(
        '<U19', '<i4', '<i4', '<U9', '<U63', '<i4', '|b1', '|b1', '<f4'),
        delimiter=",", names=True)

    status_rtu = np.genfromtxt(open(status_data_rtu, 'rb'), dtype=(
        '<U19', '<i4', '<i4', '<U9', '<U63', '<i4', '|b1', '|b1', '<f4'),
        delimiter=",", names=True)

    warning_wec = np.genfromtxt(open(warning_data_wec, 'rb'), dtype=(
        '<U19', '<i4', '<i4', '<U9', '<U63', '|b1', '<f4'),
        delimiter=",", names=True)

    warning_rtu = np.genfromtxt(open(warning_data_rtu, 'rb'), dtype=(
        '<U19', '<i4', '<i4', '<U9', '<U63', '|b1', '<f4'),
        delimiter=",", names=True)

    # Convert dates in the files to UNIX timestamps

    data_files = (SCADA, status_rtu, status_wec, warning_rtu, warning_wec)

    for data_file in data_files:
        # Convert datetimes to Unix timestamps (as strings)
        time = data_file['Time']
        for i in range(0, len(time)):
            t = dt.datetime.strptime(time[i], "%d/%m/%Y %H:%M:%S")
            t = (t - dt.datetime.fromtimestamp(3600)).total_seconds()
            time[i] = t

    # convert Unix timestamp string to float (for some reason this
    # doesn't work when in the loop above)

    dtlist = SCADA.dtype.descr
    dtlist[0] = (dtlist[0][0], '<f4')
    dtlist = np.dtype(dtlist)
    SCADA = SCADA.astype(dtlist)

    dtlist = status_wec.dtype.descr
    dtlist[0] = (dtlist[0][0], '<f4')
    dtlist = np.dtype(dtlist)
    status_wec = status_wec.astype(dtlist)

    dtlist = status_rtu.dtype.descr
    dtlist[0] = (dtlist[0][0], '<f4')
    dtlist = np.dtype(dtlist)
    status_rtu = status_rtu.astype(dtlist)

    dtlist = warning_wec.dtype.descr
    dtlist[0] = (dtlist[0][0], '<f4')
    dtlist = np.dtype(dtlist)
    warning_wec = warning_wec.astype(dtlist)

    dtlist = warning_rtu.dtype.descr
    dtlist[0] = (dtlist[0][0], '<f4')
    dtlist = np.dtype(dtlist)
    warning_rtu = warning_rtu.astype(dtlist)

    # Add 2 extra columns - Inverter_averages and Inverter_std_dev, as features
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
    means = pd.DataFrame(SCADA[inverters]).mean(axis=1).values
    stds = pd.DataFrame(SCADA[inverters]).std(axis=1).values
    SCADA = rec.append_fields(SCADA, ['Inverter_averages', 'Inverter_std_dev'],
                              data=[means, stds], usemask=False)

    return SCADA, status_wec, status_rtu, warning_wec, warning_rtu


# ---------------Filtering Functions-----------------------------
def power_curve_filtering(SCADA):
    """The algorithm, taken from [1], is used to label the data by
    filtering using the power curve. It primarily takes a SCADA
    argument, and this is then filtered according to a visual inspection
    of the power curve. The primary outputs are the filtered good and
    bad points on the power curve. All other outputs relate to plotting
    this data nicely in the power_curve_filtered_plot function.

    Parameters
    ----------
    SCADA: ndarray
        The SCADA data to be filtered. Note this must be either the
        SCADA data imported using import_data, or a subset of this data

    Returns
    -------
    SCADA_good_pc: ndarray
        The SCADA data marked as part of the nominal power curve by the
        algorithm
    SCADA_bad_pc: ndarray
        The SCADA data marked as anomalous by the algorithm
    SCADA_bin_averages: ndarray
        The average wind speed for each wind speed bin
    bins: ndarray
        The different wind speed bins
    x2: ndarray
        The x points for the generated interpolated power curve
    y2: ndarray
        The y points for the generated interpolated power curve
    upper_limit_ud:
        The upper limit of the power curve, above which points were
        marked as abnormal
    lower_limit_ud:
        The lower limit of the power curve, below which points were
        marked as abnormal

    References
    ----------
    [1] J. Park, J. Lee, K. Oh, and J. Lee, “Development of a Novel
    Power Curve Monitoring Method for Wind Turbines and Its Field
    Tests”,, IEEE Trans. Energy Convers., vol. 29, no. 1, pp. 119–128,
    2014.
    """

    # basic filtering for SCADA_real
    SCADA_real = SCADA[np.where((SCADA['Time'] > 0) &
                                (SCADA['WEC_ava_windspeed'] < 20))]

    # ------------------------------------------------------------------
    # --------------------------Algorithm loop start--------------------
    # ------------------------------------------------------------------
    SCADA_loop = SCADA_real
    # when terminate = True, the algorithm terminates
    terminate = False
    # initialise the standard deviation of the bins (empty array):
    SCADA_bin_stds_avg = np.zeros(200)
    # this increases on every loop iteration:
    k = 1

    while terminate is False:
        # --------1: set windspeed bins, width=1/loop iter no. (k)------
        max_wind = np.ceil(np.nanmax(SCADA_loop['WEC_ava_windspeed']))
        bin_width = 1 / k
        bins = np.arange(0, max_wind + bin_width, bin_width)

        # initialise average and std arrays
        SCADA_bin_averages = np.zeros(len(bins))
        SCADA_bin_stds = np.zeros(len(bins))

        # --------2: get average and std for each bin ------------------
        i = 0
        for i in range(0, len(bins)):
            SCADA_bin_averages[i] = np.mean(SCADA_loop[
                np.where((SCADA_loop['WEC_ava_windspeed'] >=
                          (bins[i] - bin_width / 2)) &
                         (SCADA_loop['WEC_ava_windspeed'] <
                          (bins[i] + bin_width / 2)))]['WEC_ava_Power'])
            SCADA_bin_stds[i] = np.std(SCADA_loop[
                np.where((SCADA_loop['WEC_ava_windspeed'] >=
                          (bins[i] - bin_width / 2)) &
                         (SCADA_loop['WEC_ava_windspeed'] <
                          (bins[i] + bin_width / 2)))]['WEC_ava_Power'])
            i = +1
        SCADA_bin_stds_avg[k] = np.nanmean(SCADA_bin_stds)

        # --------3: create splines-------------------------------------
        x = bins
        y = SCADA_bin_averages

        x2 = np.round(np.arange(0.0, 20.1, 0.1), 1)

        tck = splrep(x, y)

        tck_list = list(tck)
        yl = y.tolist()
        tck_list[1] = yl + [0.0, 0.0, 0.0, 0.0]

        y2 = splev(x2, tck_list)

        # --------4: Find left/right shifts:----------------------------

        # initialise SCADA_cur & PDL
        SCADA_cur = SCADA_loop
        PDL = np.zeros(100)
        PDL[0] = 20
        j = 0
        b_shift = .8
        dv = 0.1
        upper_limit_lr = np.zeros(len(x2))
        lower_limit_lr = np.zeros(len(x2))

        while (PDL[j] - PDL[j - 1]) >= b_shift:
            j += 1

            # create shifts
            right_shift = np.round(x2 + (dv * j), 1)
            left_shift = np.round(x2 - (dv * j), 1)

            t_inds = np.array([])

            # find the points which lie inside the upper and lower power
            # limits for the shifted power curves
            for i in range(0, len(x2)):
                # find where the upper and lower power limits are on the
                # left/right curves at the current windspeed. These are
                # the upper/lower "y" values for the corresponding
                # current x value
                upper_limit_lr[i] = np.nansum(
                    y2[np.where(left_shift == x2[i])])

                # this fixes the problem whereby the power curve is
                # shifted left, so the final few "upper_limit_lr" values
                # don't exist, so are shown as zeroes
                if (i >= 150) & (upper_limit_lr[i] == 0):
                    upper_limit_lr[i] = y2[i]

                lower_limit_lr[i] = np.nansum(
                    y2[np.where(right_shift == x2[i])])

                # get indices of points inside these lines
                t_inds_cur = np.where(
                    (SCADA_cur['WEC_ava_windspeed'] == x2[i]) &
                    (SCADA_cur['WEC_ava_Power'] >= lower_limit_lr[i]))
                t_inds = np.concatenate([t_inds, t_inds_cur[0]])

            t_inds = t_inds.astype(int)

            # make an array of these points
            mask = np.array(False).repeat(len(SCADA_cur))
            mask[t_inds] = True
            SCADA_inside_wind = SCADA_cur[mask]

            # calculate PDL
            PDL[j] = (len(SCADA_inside_wind) / len(SCADA_cur)) * 100

        # --------5: Find up/down shifts:-------------------------------

        # initialise SCADA_cur & PDL
        dP = 5
        y_offset = .03
        j = 0
        PDL = np.zeros(300)
        PDL[0] = 1
        SCADA_cur2 = SCADA_cur

        while (PDL[j] - PDL[j - 1]) >= y_offset:
            j += 1

            # create shifts
            upper_limit_ud = upper_limit_lr + (dP * j)
            lower_limit_ud = lower_limit_lr - (dP * j)

            t_inds = np.array([])

            # find the points which lie inside the upper and lower power
            # limits for the shifted power curves
            for i in range(0, len(x2)):
                # get indices of points inside these lines
                t_inds_cur = np.where(
                    (SCADA_cur2['WEC_ava_windspeed'] == x2[i]) &
                    (SCADA_cur2['WEC_ava_Power'] >= lower_limit_ud[i]))
                t_inds = np.concatenate([t_inds, t_inds_cur[0]])

            t_inds = t_inds.astype(int)

            # make an array of these points
            mask = np.array(False).repeat(len(SCADA_cur2))
            mask[t_inds] = True
            SCADA_inside_wind2 = SCADA_cur2[mask]

            # calculate PDL
            PDL[j] = (len(SCADA_inside_wind2) / len(SCADA_cur2)) * 100

        # set the output as the input of the next loop
        SCADA_loop = SCADA_inside_wind2

        # Check if the loop will be terminated
        a_loop = SCADA_bin_stds_avg[k] - SCADA_bin_stds_avg[k - 1]

        if a_loop < 1:
            terminate = True

        k += 1

    # ------------------------------------------------------------------
    # ------------------------Algorithm loop end------------------------
    # ------------------------------------------------------------------

    # list out good SCADA indices:
    SCADA_good_pc = SCADA_inside_wind2

    # list out bad SCADA indices:
    bad_mask = np.array([True]).repeat(len(SCADA_real))

    for time in SCADA_good_pc['Time']:
        bad_mask[np.where(SCADA_real['Time'] == time)] = False
    SCADA_bad_pc = SCADA_real[bad_mask]

    return SCADA_good_pc, SCADA_bad_pc, SCADA_bin_averages, bins, x2, y2,
    upper_limit_ud, lower_limit_ud


def filtering(
    SCADA, filter_file, column_name, time_diff_before=3600,
        time_diff_after=3600, good=True, *filter_codes):
    """This function filters the SCADA data obtained in import_data, or
    a subset of that data, by matching the data with the timestamps, and
    a band around the timestamps, of certain status or warning code
    messages. The end result is SCADA data which corresponds to certain
    operating states or faults.

    Parameters
    ----------
    SCADA: ndarray
        The SCADA data to be filtered. Must be the SCADA data obtained
        from import_data, or a subset of that data
    filter_file: ndarray
        The is one of: warning_rtu, warning_wec, status_rtu, status_wec.
    column_name: string
        Refers to the column being filtered (i.e. "Main_Status" or
        "Full_Status").
    time_diff_before: integer, optional
        The timeband before which to be filtered
    time_diff_after: integer, optional
        Timeband after which to be filtered
    good: Boolean
        Refers to whether the passed filter codes refer to fault data or
        fault-free data. If good=True, then the function will assume the
        filter_codes provided correspond to nominal operation, and will
        be filtered according to [time_of_status + time_diff_before,
        time_of_next_status - time_diff_after]. If good=False, it's
        assumed filter_codes refer to faults, and will be filtered
        according to[time_of_status - time_diff_before,
        time_of_next_status + time_diff_after].
    *filter_codes: array of str or int
        The set of codes to be filtered. Can be a single value or an
        array. If column_name is "Full_Status", then it must be a set of
        strings (e.g. '0 : 0' for nominal operation). Otherwise, an
        integer referring to the Main_Status.

    Returns
    -------
    SCADA_good: ndarray
        If good=True, SCADA_good is data strictly corresponding to
        fault-free data.
        If good=False, SCADA_good is data which isn't faulty data (but
        not necessarily fault-free, i.e. it could include times when the
        turbine was down for routine maintenance or curtailed power
        output, etc.).
    SCADA_bad: ndarray
        If good=True, SCADA_bad is data which isn't strictly fault-free
        (but not necessarily definitely faulty, i.e. it could include
        times when the turbine was down for routine maintenance or
        curtailed power output, etc.).
        If good=False, SCADA_bad is date strictly corresponding to fault
        data.

    """

    # Get the indices of filter_file which do NOT correspond to the
    # passed filter codes
    filter_file_indices = np.array([], dtype='i4')

    for filter_code in filter_codes:
        f = np.where((filter_file[column_name] == filter_code))
        filter_file_indices = np.sort(
            np.concatenate((filter_file_indices, f[0]), axis=0))

    # this finds SCADA timestamps which are greater than the "bad" wec
    # time less a time_diff, AND MORE than the next wec time + the
    # time_diff
    SCADA_filtered_indices = np.array([], dtype='i4')

    if filter_file[-1] == filter_file[filter_file_indices][-1]:
        # less 1 so as not to create an out of bounds error at run time
        index_range = range(0, len(filter_file_indices) - 1)
    else:
        # if it's not the last entry, we're all good
        index_range = range(0, len(filter_file_indices))

    if good is True:
        for i in index_range:
            g1 = np.where(
                (SCADA['Time'] >= filter_file['Time'][filter_file_indices[i]] +
                    time_diff_before) &
                (SCADA['Time'] <
                    filter_file['Time'][filter_file_indices[i] + 1] -
                    time_diff_after))

            SCADA_filtered_indices = np.concatenate(
                (SCADA_filtered_indices, g1[0]), axis=0)

        SCADA_filtered_indices = np.unique(SCADA_filtered_indices)

        # create the "good" mask
        mask = np.array([False]).repeat(len(SCADA))
        mask[SCADA_filtered_indices] = True
        SCADA_good = SCADA[mask]

        # create the "bad" mask
        mask = np.array([True]).repeat(len(SCADA))
        mask[SCADA_filtered_indices] = False
        SCADA_bad = SCADA[mask]

    else:
        for i in index_range:
            g1 = np.where(
                (SCADA['Time'] >= filter_file['Time'][filter_file_indices[i]] -
                    time_diff_before) &
                (SCADA['Time'] <
                    filter_file['Time'][filter_file_indices[i] + 1] +
                    time_diff_after))

            SCADA_filtered_indices = np.concatenate(
                (SCADA_filtered_indices, g1[0]), axis=0)

        SCADA_filtered_indices = np.unique(SCADA_filtered_indices)

        # create the "good" mask
        mask = np.array([True]).repeat(len(SCADA))
        mask[SCADA_filtered_indices] = False
        SCADA_good = SCADA[mask]

        # create the "bad" mask
        mask = np.array([False]).repeat(len(SCADA))
        mask[SCADA_filtered_indices] = True
        SCADA_bad = SCADA[mask]

    return SCADA_good, SCADA_bad


def get_fault_data(before, after):
    """This function is a shortcut to get a list of faults using the
    filtering() function. Returns a bunch of different faults.

    Parameters
    ----------
    before: integer
        The time_diff_before to be passed to the filtering() function
    after: integer
        The time_diff_after to be passed to the filtering() function

    Returns
    -------
    SCADA_all_faults: ndarray
        An array of all SCADA data corresponding to fault times
    SCADA_feeding_faults: ndarray
        An array of SCADA data corresponding to feeding faults
    SCADA_aircooling_faults: ndarray
        An array of SCADA data corresponding to aircooling faults
    SCADA_excitation_faults: ndarray
        An array of SCADA data corresponding to excitation faults
    SCADA_generator_heating_faults: ndarray
        An array of SCADA data corresponding to generator heating faults
    SCADA_mains_failure_faults: ndarray
        An array of SCADA data corresponding to mains_failure faults

    """
    # Shortcut Function to get all the fault data
    faults = (80, 62, 228, 60, 9)
    SCADA_all_faults = filtering(
        SCADA, status_wec, 'Main_Status', before, after, False, *faults)[1]
    SCADA_feeding_faults = filtering(
        SCADA, status_wec, 'Main_Status', before, after, False, 62)[1]
    SCADA_mains_failure_faults = filtering(
        SCADA, status_wec, 'Main_Status', before, after, False, 60)[1]
    SCADA_aircooling_faults = filtering(
        SCADA, status_wec, 'Main_Status', before, after, False, 228)[1]
    SCADA_excitation_faults = filtering(
        SCADA, status_wec, 'Main_Status', before, after, False, 80)[1]
    SCADA_generator_heating_faults = filtering(
        SCADA, status_wec, 'Main_Status', before, after, False, 9)[1]

    return SCADA_all_faults, SCADA_feeding_faults, SCADA_aircooling_faults, \
        SCADA_excitation_faults, SCADA_generator_heating_faults, \
        SCADA_mains_failure_faults

# ---------------------Exporting Data-----------------------------------


def export_data(filenames, data):
    """Export a csv of a subset of the SCADA data (e.g. relating to a
    specific fault, or fault-free)

    Parameters
    ----------
    filenames: str or array of strings
        The file name(s) to be exported
    data: ndarray or array of ndarrays
        The corresponding SCADA data to be exported
    """
    for f, d in zip(filenames, data):
        dtlist = np.array(d.dtype.descr)
        headings = ""
        for i in dtlist[:, 0]:
            headings += (i)
            headings += (",")
        headings = headings[0:-1]
        np.savetxt(
            f, d, delimiter=',', newline='\r\n', header=headings, fmt='%s')

# -------------------------Plot Functions-------------------------------

# These are various different plotting functions used for testing, etc.


def power_curve_filtered_plot(
    SCADA_good, SCADA_bad, SCADA_bin_averages, bins, x2, y2, upper_limit_ud,
        lower_limit_ud):
    """This function generates a nice plot from the data generated in
    power_curve_filtering()

    Parameters
    ----------
    SCADA_good: ndarray
        The "good" SCADA data from power_curve_filtering() to be plotted
    SCADA_bad: ndarray
        The "bad" SCADA data from power_curve_filtering() to be plotted
    SCADA_bin_averages, bins, x2, y2, upper_limit_ud, lower_limit_ud:
        These are all variables from power_curve_filtering().

    See help(power_curve_filtering) for more info, and an explanation of
    the algorithm)

    """

    # plot it all
    plt.figure(figsize=(40, 20))

    # ax1=fig.add_subplot(111)

    power_curve_plot = plt.plot(x2, y2, 'g', linewidth=3.0)

    # ----left/right plot (for testing only)----
    # ax1.scatter(SCADA_inside_wind['WEC_ava_windspeed'],
    #     SCADA_inside_wind['WEC_ava_Power'], c='g', s=50)
    # upper_limit_plot = ax1.plot(x2, upper_limit_lr, 'y', linewidth=3.0)
    # lower_limit_plot = ax1.plot(x2, lower_limit_lr, 'r', linewidth=3.0)

    # up/down plot
    # good and bad points
    ava_good_temp = (SCADA_good['CS101__Nacelle_ambient_temp_1'] +
                     SCADA_good['CS101__Nacelle_ambient_temp_2']) / 2
    ava_bad_tmep = (SCADA_bad['CS101__Nacelle_ambient_temp_1'] +
                    SCADA_bad['CS101__Nacelle_ambient_temp_2']) / 2

    good_plt = plt.scatter(
        SCADA_good['WEC_ava_windspeed'], SCADA_good['WEC_ava_Power'],
        c=ava_good_temp, cmap=plt.cm.Blues, linewidth='0', s=50)
    bad_plt = plt.scatter(
        SCADA_bad['WEC_ava_windspeed'], SCADA_bad['WEC_ava_Power'],
        c=ava_bad_tmep, cmap=plt.cm.Reds, linewidth='0', s=50)

    # upper and lower limits
    upper_limit_plot = plt.plot(x2, upper_limit_ud, 'black', linewidth=1.0)
    lower_limit_plot = plt.plot(x2, lower_limit_ud, 'black', linewidth=1.0)

    # show the bins on top!
    plt.scatter(bins, SCADA_bin_averages, c='r', label='bins', s=100)

    # put a grid on it
    plt.grid(b=True, which='major', color='b', linestyle='-')
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='r', linestyle='--')

    # legend, title, colorbar
    plt.legend(loc='upper left')
    plt.title("Filtered Power Curve")
    plt.colorbar(good_plt)

    plt.show()


def standard_plot(
    SCADA_good, SCADA_fault, title='Power Curve Plot',
        temp=False):

    # plot it all
    plt.figure(figsize=(40, 20))

    # up/down plot
    # good and bad points
    if temp is True:
        ava_good_temp = (SCADA_good['CS101__Nacelle_ambient_temp_1'] +
                         SCADA_good['CS101__Nacelle_ambient_temp_2']) / 2
        ava_fault_temp = (SCADA_fault['CS101__Nacelle_ambient_temp_1'] +
                          SCADA_fault['CS101__Nacelle_ambient_temp_2']) / 2
        good_colour = ava_good_temp
        bad_colour = ava_fault_temp
    else:
        good_colour = 'b'
        bad_colour = 'r'

    good_plt = plt.scatter(
        SCADA_good['WEC_ava_windspeed'], SCADA_good['WEC_ava_Power'],
        c=good_colour, cmap=plt.cm.Blues, linewidth='0', s=50)
    fault_plt = plt.scatter(
        SCADA_fault['WEC_ava_windspeed'], SCADA_fault['WEC_ava_Power'],
        c=bad_colour, cmap=plt.cm.Reds, linewidth='0', s=50)

    # put a grid on it
    plt.grid(b=True, which='major', color='b', linestyle='-')
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='r', linestyle='--')

    # legend, title, colorbar
    plt.legend(loc='upper left')
    plt.title(title)

    if temp:
        plt.colorbar(good_plt)

    plt.show()


# --------------------SVM Functions-------------------------------------

def generate_labels(
    no_fault_data, fault_data, features, normalize=True,
        split=0.2):
    """Generate labels for the SCADA data.

    Parameters
    ----------
    no_fault_data: ndarray
        Subset of SCADA data imported using import_data which
        corresponds to fault-free data
    fault_data: ndarray
        Subset of SCADA data imported using import_data which
        corresponds to fault data. Examples include SCADA_all_faults,
        SCADA_feeding_faults, etc.
    features: array of strings
        set of features used in the dataset.
    normalize: Boolean, optional
        Whether or not to normalize the training data. Default is True.
    split: float, optional
        The ratio of testing : training data to use. Default is 0.2

    Returns
    -------
    X_train: ndarray
        The set of data to be trained on
    y_train: ndarray
        The associated labels
    X_test: ndarray
        The set of data to be tested on
    y_test: ndarray
        The associated labels
    X_train_bal: ndarray
        Used for balanced training data (i.e. no. fault class=no.
        of fault-free class)
    y_train_bal: ndarray
        Used for balanced testing data
    """

    good_labels = np.zeros(len(no_fault_data), dtype=np.int)
    bad_labels = np.ones(len(fault_data), dtype=np.int)

    # append the appropriate labels to the data
    good = rec.append_fields(
        no_fault_data[features], ['label'], data=[good_labels], usemask=False)
    bad = rec.append_fields(
        fault_data[features], ['label'], data=[bad_labels], usemask=False)

    # join all the data together and shuffle it
    dataset = np.concatenate([good, bad])
    np.random.shuffle(dataset)

    # separate the training data from the labels, and normalize
    y = dataset['label']
    X = rec.drop_fields(dataset, ['label'], False).view(
        np.float32).reshape(len(dataset), len(dataset.dtype) - 1)

    # Create Training and Test Sets
    if normalize is True:
        X_norm = prep.normalize(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X_norm, y, test_size=split)
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=split)

    # Create the balanced training sets
    X_train_bad = X_train[np.where(y_train == 1)]
    y_train_bad = y_train[np.where(y_train == 1)]

    X_train_good_bal = X_train[
        np.where(y_train == 0)][0:round(len(X_train_bad))]
    y_train_good_bal = y_train[
        np.where(y_train == 0)][0:round(len(X_train_bad))]

    X_train_bal_unshuffled = np.concatenate([X_train_good_bal, X_train_bad])
    y_train_bal_unshuffled = np.concatenate([y_train_good_bal, y_train_bad])

    balanced_training_data = np.append(
        X_train_bal_unshuffled, np.array([y_train_bal_unshuffled]).T, axis=1)
    np.random.shuffle(balanced_training_data)
    y_train_bal = balanced_training_data[:, 29]
    X_train_bal = balanced_training_data[:, 0:29]

    return X_train, X_test, y_train, y_test, X_train_bal, y_train_bal
