import os
import sys
import csv

import tensorflow as tf

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator, STORE_EVERYTHING_SIZE_GUIDANCE


def tabulate_events(dir):
    summary_iterator = EventAccumulator(dir, size_guidance=STORE_EVERYTHING_SIZE_GUIDANCE).Reload()
    
    tag = 'epoch/avg_return'

    steps = [e.step for e in summary_iterator.Tensors(tag)]

    values= [tf.make_ndarray(e.tensor_proto).item() for e in summary_iterator.Tensors(tag)]

    return zip(steps, values)

def to_csv(dpath, is_parent=True):
    if is_parent:
        dirs = os.listdir(dpath)
        for dir in dirs:
            data = tabulate_events(os.path.join(dpath, dir))
            write_csv(data, get_file_path(dir, dpath))
    else:
        data = tabulate_events(dpath)
        write_csv(data, get_file_path(dpath))

def write_csv(data, path):
    with open(path, 'w') as file:
        wr = csv.writer(file)
        wr.writerow(['step', 'value'])
        wr.writerows(data) 

def get_file_path(dir, dpath = None):
    file_name = f'{os.path.basename(os.path.normpath(dir))}.csv'
    if dpath:
        folder_path = os.path.join('data', os.path.basename(os.path.abspath(dpath)))
    else: 
        folder_path = os.path.abspath(os.path.join('data', dir, '..'))
        
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    return os.path.join(folder_path, file_name)


if __name__ == '__main__':
    if len(sys.argv) == 1 or len(sys.argv) > 3:
        exit('usage: python get_csv.py <path_to_dir> [parent](optional)')
    elif len(sys.argv) == 3:
        is_parent = 'parent' == sys.argv[2]
    else:
        is_parent = False

    to_csv(sys.argv[1], is_parent=is_parent)