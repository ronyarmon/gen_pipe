import re
import os
import numpy as np

def get_runs_counter(main_dir_path):
    '''
    Return counter for the directory holding the run results
    For example, if the run results directories are run1, run2
    the function will return the number 3, to be used in naming the directory
    of the next run
    :params:
    dir_path: the path to the directory storing all runs
    '''

    num = 0
    results_dirs = next(os.walk(main_dir_path))[1]
    run_names = [d for d in results_dirs\
                 if (('run' in d) & bool(re.findall('(\d{1,})', d)))]
    if run_names:
        run_nums = []
        for rd in run_names:
            ext_num = re.findall('(\d{1,})', rd)
            ext_num = int(ext_num[0])
            run_nums.append(ext_num)
        num = sorted(run_nums, reverse=True)[0]
    return str(num+1)

def build_name_metadata(metadata_file):

    '''
    Build the name for run results using the details in a metadata file
    :param metadata_file
    '''
    metadata_dict = np.load(metadata_file, allow_pickle=True)[()]
    run_name = ''
    for k, v in metadata_dict.items():
        if int(v) > 9999:
            v = '{v:.2e}'.format(v=v)
        run_name = '{r}{v}{k}_'.format(r=run_name, k=k, v=v)

    return run_name.replace(' ', '_').replace('+', '')

def build_run_dir_name(run_dir_name, main_dir_path):

    '''
    Build the name for a directory to store run results
    :params:
    run_dir_name: The directory name given by the user.
    main_dir: The path in which the stored directory is being stored.
    :dependencies: run_metadata.txt, a file into which the run metadata is written
    and is used if the user specifies 'm'

    :return:
    '''

    if run_dir_name:
        if run_dir_name == 'm':
            run_dir_name = build_name_metadata('metadata.npy')
        else:
            run_dir_name = run_dir_name.replace(' ', '_')

    # If the user does not specify a name build a name in the form of run+count
    else:
        next_run_count = get_runs_counter(main_dir_path)
        run_dir_name = 'run{n}'.format(n=next_run_count)

    return run_dir_name.rstrip('_')
