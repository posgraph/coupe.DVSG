#from utils import *
import collections
import numpy as np
import tensorlayer as tl
import cv2
import os
from threading import Thread
from threading import Lock
from datetime import datetime

class Data_Loader:
    def __init__(self, config, is_train, thread_num = 3):

        self.config = config
        self.is_train = is_train
        self.thread_num = thread_num

        self.num_partition = 2
        self.skip_length = np.array(self.config.skip_length)

        self.stab_folder_path_list, self.stab_file_path_list, self.num_files = self._load_file_list(self.config.stab_path)
        self.unstab_folder_path_list, self.unstab_file_path_list, _ = self._load_file_list(self.config.unstab_path)
        self.of_folder_path_list, self.of_frame_path_list, _ = self._load_file_list(config.of_path)
        self.surf_folder_path_list, self.surf_frame_path_list, _ = self._load_file_list(config.surf_path)

        self.h = self.config.height
        self.w = self.config.width
        self.batch_size = self.config.batch_size

    def init_data_loader(self, network, is_pretrain):
        self.sample_num = network.sample_num

        self.idx_video = []
        self.idx_frame = []
        self.init_idx()

        self.num_itr = int(np.ceil(len(sum(self.idx_frame, [])) / self.batch_size))

        self.lock = Lock()
        self.is_end = False

        ### THREAD HOLDERS ###
        self.net_placeholder_names = None
        self.is_pretrain_input = False
        self.network = network
        if is_pretrain:
            self.net_placeholder_names = list(network.inputs_pretrain.keys())
            self.net_inputs = network.inputs_pretrain
            self.is_pretrain_input = True
        else:
            self.net_placeholder_names = list(network.inputs.keys())
            self.net_inputs = network.inputs
            self.is_pretrain_input = False
        self.threads = [None] * self.thread_num
        self.threads_unused = [None] * self.thread_num
        self.feed_dict_holder = self._set_feed_dict_holder(self.net_placeholder_names, self.thread_num)
        self._init_data_thread()

        #self._print_log()

    def init_idx(self):
        self.idx_video = []
        self.idx_frame = []
        for i in range(len(self.stab_file_path_list)):
            total_frames = len(self.stab_file_path_list[i])

            self.idx_frame.append(list(range(0, total_frames - (self.skip_length[-1] - self.skip_length[0] + 1) - (self.num_partition - 1))))
            for j in np.arange(self.sample_num - 1):
                self.idx_frame[i].append(0)
            self.idx_video.append(i)

        self.is_end = False

    def reset_to_train_input(self, network):
        self.is_pretrain_input = False
        self.network = network
        self.net_placeholder_names = network.inputs.keys()
        self.net_inputs = network.inputs
        for thread_idx in np.arange(self.thread_num):
            if self.threads[thread_idx].is_alive():
                self.threads[thread_idx].join()
        self.threads = [None] * self.thread_num
        self.threads_unused = [None] * self.thread_num
        self.feed_dict_holder = self._set_feed_dict_holder(self.net_placeholder_names, self.thread_num)
        self._init_data_thread()

    def get_batch(self, threads_unused, thread_idx):
        assert(self.net_placeholder_names is not None)
        #tl.logging.debug('\tthread[%s] > get_batch start [%s]' % (str(thread_idx), str(datetime.now())))

        ## random sample frame indexes
        self.lock.acquire()
        #tl.logging.debug('\t\tthread[%s] > acquired lock [%s]' % (str(thread_idx), str(datetime.now())))

        if self.is_end:
            #tl.logging.debug('\t\tthread[%s] > releasing lock 1 [%s]' % (str(thread_idx), str(datetime.now())))
            self.lock.release()
            return

        video_idxes = []
        frame_offsets = []

        actual_batch = 0
        for i in range(0, self.batch_size):
            if i == 0 and len(self.idx_video) == 0:
                self.is_end = True
                #tl.logging.debug('\t\tthread[%s] > releasing lock 2 [%s]' % (str(thread_idx), str(datetime.now())))
                self.lock.release()
                return
            elif i > 0 and len(self.idx_video) == 0:
                break

            else:
                if self.is_train:
                    idx_x = np.random.randint(len(self.idx_video))
                    video_idx = self.idx_video[idx_x]
                    idx_y = np.random.randint(len(self.idx_frame[video_idx]))
                else:
                    idx_x = 0
                    idx_y = 0
                    video_idx = self.idx_video[idx_x]

            frame_offset = self.idx_frame[video_idx][idx_y]
            video_idxes.append(video_idx)
            frame_offsets.append(frame_offset)
            self._update_idx(idx_x, idx_y)
            actual_batch += 1

        #tl.logging.debug('\t\tthread[%s] > releasing lock 4 [%s]' % (str(thread_idx), str(datetime.now())))
        self.lock.release()
        threads_unused[thread_idx] = True

        ## init thread lists
        data_holder = self._set_data_holder(self.net_placeholder_names, actual_batch)

        ## start thread
        threads = [None] * actual_batch
        for batch_idx in range(actual_batch):
            video_idx = video_idxes[batch_idx]
            frame_offset = frame_offsets[batch_idx]
            threads[batch_idx] = Thread(target = self.read_dataset, args = (data_holder, batch_idx, video_idx, frame_offset))
            threads[batch_idx].start()

        for batch_idx in range(actual_batch):
            threads[batch_idx].join()

        surfs_t_1 = data_holder['surfs_t_1']
        surfs_t_1 = np.hstack(surfs_t_1)
        surfs_t_1_padded = np.ones((actual_batch, 2, max(data_holder['surfs_dim_t_1']), 2))
        surfs_t_1_padded[:, 0, :, :] = surfs_t_1_padded[:, 0, :, :] * 0
        surfs_t_1_padded[:, 1, :, 0] = surfs_t_1_padded[:, 1, :, 0] * 0
        surfs_t_1_padded[:, 1, :, 1] = surfs_t_1_padded[:, 1, :, 1] * self.h
        mask_surfs_t_1 = np.zeros((actual_batch, 2, max(data_holder['surfs_dim_t_1']), 2))
        for i in np.arange(len(data_holder['surfs_dim_t_1'])):
            mask_surfs_t_1[i, :, 0:data_holder['surfs_dim_t_1'][i], :] = 1
        mask_surfs_t_1 = (mask_surfs_t_1 == 1)
        surfs_t_1_padded[mask_surfs_t_1] = surfs_t_1
        data_holder['surfs_t_1'] = surfs_t_1_padded 

        surfs_t = data_holder['surfs_t']
        surfs_t = np.hstack(surfs_t)
        surfs_t_padded = np.ones((actual_batch, 2, max(data_holder['surfs_dim_t']), 2))
        surfs_t_padded[:, 0, :, :] = surfs_t_padded[:, 0, :, :] * 0
        surfs_t_padded[:, 1, :, 0] = surfs_t_padded[:, 1, :, 0] * 0
        surfs_t_padded[:, 1, :, 1] = surfs_t_padded[:, 1, :, 1] * self.h
        mask_surfs_t = np.zeros((actual_batch, 2, max(data_holder['surfs_dim_t']), 2))
        for i in np.arange(len(data_holder['surfs_dim_t'])):
            mask_surfs_t[i, :, 0:data_holder['surfs_dim_t'][i], :] = 1
        mask_surfs_t = (mask_surfs_t == 1)
        surfs_t_padded[mask_surfs_t] = surfs_t
        data_holder['surfs_t'] = surfs_t_padded

        for (key, val) in data_holder.items():
            if 'surf' not in key:
                data_holder[key] = np.concatenate(data_holder[key][0 : actual_batch], axis = 0)

        for holder_name in self.net_placeholder_names:
            self.feed_dict_holder[holder_name][thread_idx] = data_holder[holder_name]

        #tl.logging.debug('\tthread[%s] > get_batch done [%s]' % (str(thread_idx), str(datetime.now())))

    def read_dataset(self, data_holder, batch_idx, video_idx, frame_offset):
        #sampled_frame_idx = np.arange(frame_offset, frame_offset + self.sample_num * self.skip_length, self.skip_length)
        sampled_frame_idx = frame_offset + self.skip_length

        patches_t_1_temp = [None] * len(sampled_frame_idx)
        patches_t_temp = [None] * len(sampled_frame_idx)

        threads = [None] * len(sampled_frame_idx)
        for frame_idx in range(len(sampled_frame_idx)):

            is_read_stab = False if frame_idx == len(sampled_frame_idx) - 1 else True 

            sampled_idx = sampled_frame_idx[frame_idx]
            threads[frame_idx] = Thread(target = self.read_frame_data, args = (data_holder, batch_idx, video_idx, frame_idx, sampled_idx, patches_t_1_temp, patches_t_temp, is_read_stab))
            threads[frame_idx].start()

        for frame_idx in range(len(sampled_frame_idx)):
            threads[frame_idx].join()

        patches_t_1_temp = np.concatenate(patches_t_1_temp[0: len(patches_t_1_temp)], axis = 3)
        patches_t_temp = np.concatenate(patches_t_temp[0: len(patches_t_temp)], axis = 3)

        data_holder['patches_t_1'][batch_idx] = patches_t_1_temp
        data_holder['patches_t'][batch_idx] = patches_t_temp


    def read_frame_data(self, data_holder, batch_idx, video_idx, frame_idx, sampled_idx, patches_t_1_temp, patches_t_temp, is_read_stab):
        sample_idx_t_1 = sampled_idx
        sample_idx_t = sampled_idx + 1
        # read stab frame
        stab_file_path = self.stab_file_path_list[video_idx]
        unstab_file_path = self.unstab_file_path_list[video_idx]
        of_file_path = self.of_frame_path_list[video_idx]
        surf_file_path = self.surf_frame_path_list[video_idx]

        stab_frame_t_1 = self._read_frame(stab_file_path[sample_idx_t_1])
        stab_frame_t = self._read_frame(stab_file_path[sample_idx_t])

        assert(self._get_folder_name(unstab_file_path[sample_idx_t]) == self._get_folder_name(stab_file_path[sample_idx_t]) == self._get_folder_name(surf_file_path[sample_idx_t]) == self._get_folder_name(of_file_path[sample_idx_t - 1]))
        assert(self._get_base_name(unstab_file_path[sample_idx_t]) == self._get_base_name(stab_file_path[sample_idx_t]) == self._get_base_name(surf_file_path[sample_idx_t]) == self._get_base_name(of_file_path[sample_idx_t - 1]))
        if is_read_stab:
            patches_t_1_temp[frame_idx] = stab_frame_t_1
            patches_t_temp[frame_idx] = stab_frame_t
        else:
            # read unstab frames
            unstab_frame_t_1 = self._read_frame(unstab_file_path[sample_idx_t_1])
            unstab_frame_t = self._read_frame(unstab_file_path[sample_idx_t])

            patches_t_1_temp[frame_idx] = stab_frame_t_1
            patches_t_temp[frame_idx] = unstab_frame_t

            data_holder['s_t_1_gt'][batch_idx] = stab_frame_t_1
            data_holder['s_t_gt'][batch_idx] = stab_frame_t
            data_holder['u_t_1'][batch_idx] = unstab_frame_t_1
            data_holder['u_t'][batch_idx] = unstab_frame_t

            # read optical flow
            data_holder['of_t'][batch_idx] = np.expand_dims(cv2.resize(np.load(of_file_path[sample_idx_t]), (self.w, self.h)), axis = 0) * [self.w, self.h]

            # read surf
            surfs_t_1 = self._read_surf(surf_file_path[sample_idx_t_1])
            surfs_t = self._read_surf(surf_file_path[sample_idx_t])

            data_holder['surfs_t_1'][batch_idx] = np.squeeze(surfs_t_1.flatten())
            data_holder['surfs_t'][batch_idx] = np.squeeze(surfs_t.flatten())

            data_holder['surfs_dim_t_1'][batch_idx] = np.array(surfs_t_1.shape[2]) if surfs_t_1.shape[2] != 0 else np.array(0)
            data_holder['surfs_dim_t'][batch_idx] = np.array(surfs_t.shape[2]) if surfs_t.shape[2] != 0 else np.array(0)

    def _center_crop_and_resize(self, image, size):
        shape = image.shape[:2]
        crop_size = min(shape)

        image = tl.prepro.crop(image, wrg = crop_size - 1, hrg = crop_size - 1, is_random = False)
        image = tl.prepro.imresize(image, size = size)

        return image

    def _update_idx(self, idx_x, idx_y):
        video_idx = self.idx_video[idx_x]
        del(self.idx_frame[video_idx][idx_y])

        if len(self.idx_frame[video_idx]) == 0:
            del(self.idx_video[idx_x])
            # if len(self.idx_video) != 0:
            #     self.video_name = os.path.basename(self.stab_file_path_list[self.idx_video[0]])

    def _load_file_list(self, root_path):
        folder_paths = []
        file_names = []
        num_files = 0
        for root, dirnames, filenames in os.walk(root_path):
            if len(dirnames) == 0:
                folder_paths.append(root)
                for i in np.arange(len(filenames)):
                    filenames[i] = os.path.join(root, filenames[i])
                file_names.append(np.array(sorted(filenames)))
                num_files += len(filenames)

        folder_paths = np.array(folder_paths)
        file_names = np.array(file_names)

        sort_idx = np.argsort(folder_paths)
        folder_paths = folder_paths[sort_idx]
        file_names = file_names[sort_idx]

        return np.squeeze(folder_paths), np.squeeze(file_names), np.squeeze(num_files)

    def _read_frame(self, path):
        frame = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) / 255.
        frame = cv2.resize(frame, (self.w, self.h))
        return np.expand_dims(frame, axis = 0)

    def _read_surf(self, file_path):
        rawdata = np.loadtxt(file_path)

        if len(rawdata.shape) == 2:
            output = np.zeros((2,rawdata.shape[0],2))
            for i in range(rawdata.shape[0]):
                output[0,i,0] = int(np.round(rawdata[i,0]))
                output[0,i,1] = int(np.round(rawdata[i,1]))
                output[1,i,0] = int(np.round(rawdata[i,2]))
                output[1,i,1] = int(np.round(rawdata[i,3]))

            return np.expand_dims(output - 1, axis = 0)
        else:
            return np.ones((1, 2, 0, 2)) * self.w * self.h

    def _get_base_name(self, path):
        return os.path.basename(path.split('.')[0])

    def _get_folder_name(self, path):
        path = os.path.dirname(path)
        return path.split(os.sep)[-1]

    def _set_feed_dict_holder(self, holder_names, thread_num):
        feed_dict_holder = collections.OrderedDict()
        for holder_name in holder_names:
            feed_dict_holder[holder_name] = [None] * thread_num

        return feed_dict_holder

    def _set_data_holder(self, net_placeholder_names, batch_num):
        data_holder = collections.OrderedDict()
        for holder_name in net_placeholder_names:
            data_holder[holder_name] = [None] * batch_num

        return data_holder

    def _init_data_thread(self):
        self.init_idx()
        #tl.logging.debug('INIT_THREAD [%s]' % str(datetime.now()))
        for thread_idx in range(0, self.thread_num):
            self.threads[thread_idx] = Thread(target = self.get_batch, args = (self.threads_unused, thread_idx))
            self.threads_unused[thread_idx] = False
            self.threads[thread_idx].start()

        #tl.logging.debug('INIT_THREAD DONE [%s]' % str(datetime.now()))

    def feed_the_network(self):
        thread_idx, is_end = self._get_thread_idx()
        #tl.logging.debug('THREAD[%s] > FEED_THE_NETWORK [%s]' % (str(thread_idx), str(datetime.now())))
        if is_end:
            return None, is_end

        feed_dict = collections.OrderedDict()
        for (key, val) in self.net_inputs.items():
            feed_dict[val] = self.feed_dict_holder[key][thread_idx]

        #tl.logging.debug('THREAD[%s] > FEED_THE_NETWORK DONE [%s]' % (str(thread_idx), str(datetime.now())))
        return feed_dict, is_end

    def _get_thread_idx(self):
        for thread_idx in np.arange(self.thread_num):
            if self.threads[thread_idx].is_alive() == False and self.threads_unused[thread_idx] == False:
                    self.threads[thread_idx] = Thread(target = self.get_batch, args = (self.threads_unused, thread_idx))
                    self.threads[thread_idx].start()

        while True:
            is_unused_left = False
            for thread_idx in np.arange(self.thread_num):
                if self.threads_unused[thread_idx]:
                    is_unused_left = True
                    if self.threads[thread_idx].is_alive() == False:
                        self.threads_unused[thread_idx] = False
                        return thread_idx, False

            if is_unused_left == False and self.is_end:
                self._init_data_thread()
                return None, True

    def _print_log(self):
        print('stab_folder_path_list')
        print(len(self.stab_folder_path_list))

        print('stab_file_path_list')
        total_file_num = 0
        for file_path in self.stab_file_path_list:
            total_file_num += len(file_path)
        print(total_file_num)

        print('unstab_file_path_list')
        total_file_num = 0
        for file_path in self.unstab_file_path_list:
            total_file_num += len(file_path)
        print(total_file_num)

        print('of_file_path_list')
        total_file_num = 0
        for file_path in self.of_frame_path_list:
            total_file_num += len(file_path)
        print(total_file_num)

        print('surf_file_path_list')
        total_file_num = 0
        for file_path in self.surf_frame_path_list:
            total_file_num += len(file_path)
        print(total_file_num)

        print('num itr per epoch')
        print(self.num_itr)
