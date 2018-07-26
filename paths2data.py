# -------------------------------------------------------- #
# make a list of all data paths
import glob
import numpy as np
import pandas as pd
# import logging
import os
# from multiprocessing import Pool


def find_lrs_relative_paths(dir, extension):
    def get_user_video(path):
        return [path[-2], path[-1][:-4]]
    file_paths = glob.glob(dir + '/**/*.' + extension, recursive=True)
    file_paths = [get_user_video(path.split(os.sep)) for path in file_paths]
    return pd.DataFrame(data=file_paths, columns=['person_id', 'video_id'])


def find_absolute_paths(dir, extension):
    def get_user_video(path):
        return [path[-2], path[-1][:-4]]
    file_paths = glob.glob(dir + '/**/*.' + extension, recursive=True)
    # file_paths = [get_user_video(path.split(os.sep)) for path in file_paths]
    return np.array(file_paths)  # pd.DataFrame(data=file_paths, columns=['person_id', 'video_id'])


dir = "/home/mat10/Documents/MSc_Machine_Learning/MSc_Project/audio_to_3dvideo/Data"
data_info_dir = "/home/mat10/Documents/MSc_Machine_Learning/MSc_Project/audio_to_3dvideo/Data"
extension = "tfrecords"
paths = find_absolute_paths(dir, extension)
np.savetxt(data_info_dir + "/example_set.csv", paths, delimiter=",", fmt="%s")


# Uncomment below and change to mak run on multiple cores
# def split_list(list_, num_chunks):
#     chunk_size = len(list_) // num_chunks
#     # remainder = len(list_) % num_chunks
#     res = []
#     for i in range(num_chunks):
#         if i == num_chunks-1:
#             res.append(list_[i*chunk_size:])
#         else:
#             res.append(list_[i*chunk_size:(i+1)*chunk_size])
#     return res
#
#
# if __name__ == "__main__":
#
#     # USER INPUT ---------------------------------------------------------------------------------#
#     data_root_dir = "/home/mat10/Documents/MSc_Machine_Learning/MSc_Project/Data/LRW/LRW_test"  # "/data/mat10/datasets/LRW/Data"
#     log_root_dir = data_root_dir  # "/data/mat10/datasets/LRW/Logs"
#     save_dir = data_root_dir # "/data/mat10/datasets/LRW/Paths"
#
#     extension = 'tfrecords'
#
#     datasets = ['train', 'val', 'test']
#
#     logfilename = "data_transformation_errors.log"
#
#     num_cores = 8
#     # -------------------------------------------------------------------------------------------#
#
#     LOG_FILENAME = log_root_dir + "/" + logfilename
#
#     words = os.listdir(data_root_dir)
#
#     logging.basicConfig(filename=LOG_FILENAME, level=logging.DEBUG)
#
#     words_split = split_list(words, num_cores)
#
#     logging.info('Start gathering paths')
#
#     pool = Pool(num_cores)
#     paths = pool.map(find_paths_lrw, zip(words_split, num_cores*[extension]))
#
#     logging.info('Gathering paths completed')
#
#     paths = pd.concat(paths, axis=0)
#
#     #unwords = pd.DataFrame(data=paths['word'].unique(), columns=['word'])
#     #unwords['class'] = np.arange(unwords.shape[0])
#
#     #paths = paths.merge(unwords, on=['word'], how='left')
#     train_paths = paths[paths['dataset'] == 'train']
#     val_paths = paths[paths['dataset'] == 'val']
#     test_paths = paths[paths['dataset'] == 'test']
#
#     logging.info('Saving results')
#     paths.to_csv(save_dir + "/" + "data_info_" + extension + ".csv", index=False)
#     train_paths.to_csv(save_dir + "/" + "train_data_info_" + extension + ".csv", index=False)
#     val_paths.to_csv(save_dir + "/" + "val_data_info_" + extension + ".csv", index=False)
#     test_paths.to_csv(save_dir + "/" + "test_data_info_" + extension + ".csv", index=False)
#
#     logging.info('Saving results finished')