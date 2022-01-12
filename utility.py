import numpy as np

year_bins = [2001, 2006, 2011, 2016, 2021]
age_bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 111]
txt_dict = {
    'yr_range_warning': 'Due to accuracy, we only provide the data recorded between 2001 and 2020. ',
    'age_range_warning': 'The age range you are querying is out of what we have in registry data. ',
    'success': 'Great, we have the data you need! ',
    'ep_failed_to_find': 'Sorry, we cannot find any disease name from your question. '
}


def find_age_range(age_start, age_end, txt):
    if age_start > age_end:
        age = age_start
        age_start = age_end
        age_end = age

    if age_start < age_bins[0]:  # seems no need?
        a_start = 0
        txt += txt_dict['age_range_warning']
    elif age_start >= age_bins[-1]:
        a_start = 100
        txt += txt_dict['age_range_warning']
    else:
        a_start = age_bins[np.digitize(age_start, age_bins) - 1]

    if age_end < age_bins[0]:
        a_end = 9
        txt += txt_dict['yr_range_warning']
    elif age_end >= age_bins[-1]:
        a_end = 111
        txt += txt_dict['yr_range_warning']
    else:
        a_end = age_bins[np.digitize(age_end, age_bins)] - 1

    return ' AND CAST(age_start AS INT) >= '+str(a_start)+' AND CAST(age_end AS INT) <= '+str(a_end), txt

def find_yr_range(yr_start, yr_end, txt):
    if yr_start > yr_end:
        yr = yr_start
        yr_start = yr_end
        yr_end = yr

    if yr_start < year_bins[0]:
        yr_start = 2001
        txt += txt_dict['yr_range_warning']
    elif yr_start >= year_bins[-1]:
        yr_start = 2016
        txt += txt_dict['yr_range_warning']
    else:
        yr_start = year_bins[np.digitize(yr_start, year_bins) - 1]

    if yr_end < year_bins[0]:
        yr_end = 2005
        txt += txt_dict['yr_range_warning']
    elif yr_end >= year_bins[-1]:
        yr_end = 2020
        txt += txt_dict['yr_range_warning']
    else:
        yr_end = year_bins[np.digitize(yr_end, year_bins)] - 1

    return ' AND yr_start >= ' + str(yr_start) + ' AND yr_end <= ' + str(yr_end), txt
