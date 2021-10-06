import sys
modules_path = '/Users/rarmon/Code/modules'
sys.path.append(modules_path)
from libraries import *
from utils.directories import *
project_files_dirs = os.listdir()


dataset_dir, results_dir = ('./dataset', './results')
print('results_dir:', results_dir)

# Run directory name
run_dir_name = input('run results directory name?(m=from metadata))')
print('input run_dir_name:', run_dir_name)

run_dir_name = build_run_dir_name(run_dir_name, results_dir)
run_dir_path = os.path.join(results_dir, run_dir_name)
os.mkdir(run_dir_path)
images_dir = os.path.join(run_dir_path, 'images')
os.mkdir(images_dir)
tables_dir = os.path.join(run_dir_path, 'tables')
os.mkdir(tables_dir)