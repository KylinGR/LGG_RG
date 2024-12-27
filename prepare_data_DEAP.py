# This is the processing script of DEAP dataset

import _pickle as cPickle

from train_model import *
from scipy import signal


class PrepareData:
    def __init__(self, args):
        # init all the parameters here
        # arg contains parameter settings
        self.args = args
        self.data = None
        self.label = None
        self.model = None
        self.data_path = args.data_path
        self.original_order = [
            'Fpz', 'Fp1', 'Fp2', 'AF3', 'AF4', 'AF7', 'AF8',
            'Fz', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8',
            'FCz', 'FC1', 'FC2', 'FC3', 'FC4', 'FC5', 'FC6', 'FT7', 'FT8',
            'Cz', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'T7', 'T8',
            'CP1', 'CP2', 'CP3', 'CP4', 'CP5', 'CP6', 'TP7', 'TP8',
            'Pz', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8',
            'POz', 'PO3', 'PO4', 'PO5', 'PO6', 'PO7', 'PO8',
            'Oz', 'O1', 'O2']
        self.graph_fro_DEAP = [['Fp1', 'AF3'], ['Fp2', 'AF4'], ['F3', 'F7'], ['F4', 'F8'],
                               ['Fz'],
                               ['FC5', 'FC1'], ['FC6', 'FC2'], [
                                   'C3', 'Cz', 'C4'], ['CP5', 'CP1', 'CP2', 'CP6'],
                               ['P7', 'P3', 'Pz', 'P4', 'P8'], [
                                   'PO3', 'PO4'], ['O1', 'Oz', 'O2'],
                               ['T7'], ['T8']]
        self.graph_gen_DEAP = [['Fp1', 'Fp2'], ['AF3', 'AF4'], ['F3', 'F7', 'Fz', 'F4', 'F8'],
                               ['FC5', 'FC1', 'FC6', 'FC2'], ['C3', 'Cz', 'C4'], [
                                   'CP5', 'CP1', 'CP2', 'CP6'],
                               ['P7', 'P3', 'Pz', 'P4', 'P8'], [
                                   'PO3', 'PO4'], ['O1', 'Oz', 'O2'],
                               ['T7'], ['T8']]
        self.graph_hem_DEAP = [['Fp1', 'AF3'], ['Fp2', 'AF4'], ['F3', 'F7'], ['F4', 'F8'],
                               ['Fz', 'Cz', 'Pz', 'Oz'],
                               ['FC5', 'FC1'], ['FC6', 'FC2'], ['C3'], [
                                   'C4'], ['CP5', 'CP1'], ['CP2', 'CP6'],
                               ['P7', 'P3'], ['P4', 'P8'], ['PO3', 'O1'], ['PO4', 'O2'], ['T7'], ['T8']]
        self.TS = [
            'Fpz', 'Fp1', 'Fp2', 'AF3', 'AF4', 'AF7', 'AF8',
            'Fz', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8',
            'FCz', 'FC1', 'FC2', 'FC3', 'FC4', 'FC5', 'FC6', 'FT7', 'FT8',
            'Cz', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'T7', 'T8',
            'CP1', 'CP2', 'CP3', 'CP4', 'CP5', 'CP6', 'TP7', 'TP8',
            'Pz', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8',
            'POz', 'PO3', 'PO4', 'PO5', 'PO6', 'PO7', 'PO8',
            'Oz', 'O1', 'O2']
        self.graph_type = args.graph_type

    def run(self, subject_list, split=False, expand=True):
        """
        Parameters
        ----------
        subject_list: the subjects need to be processed
        split: (bool) whether to split one trial's data into shorter segment
        expand: (bool) whether to add an empty dimension for CNN

        Returns
        -------
        The processed data will be saved './data_<data_format>_<dataset>_<label_type>/sub0.hdf'
        """
        for sub in subject_list:
            data_, label_ = self.load_data_per_subject(sub)

            if expand:
                data_ = np.expand_dims(data_, axis=-3)

            if split:
                data_, label_ = self.split(
                    data=data_, label=label_, segment_length=self.args.segment,
                    overlap=self.args.overlap, sampling_rate=self.args.sampling_rate)

            print('Data and label prepared!')
            print('data:' + str(data_.shape) + ' label:' + str(label_.shape))
            print('----------------------')
            self.save(data_, label_, sub)

    def load_data_per_subject(self, sub):
        """
        This function loads the target subject's original file
        Parameters
        ----------
        sub: which subject to load

        Returns
        -------
        data: (40, 32, 7680) label: (40, 4)
        """
        sub += 1
        sub_code = f's{"0" if sub < 10 else ""}{sub}.mat'
        subject_path = os.path.join(self.data_path, sub_code)

        with h5py.File(subject_path, 'r') as mat_file:
            data = mat_file['data'][:]
            label = mat_file['label'][:]
            
            # Reshape data to (trials, channels, samples)
            data = data.transpose(2, 1, 0)
            label = label.reshape(-1, 1)
            # Reorder channels if needed

        data = self.reorder_channel(data=data, graph=self.graph_type)
        return data, label


    def reorder_channel(self, data, graph):
        """
        This function reorder the channel according to different graph designs
        Parameters
        ----------
        data: (trial, channel, data)
        graph: graph type

        Returns
        -------
        reordered data: (trial, channel, data)
        """
        if graph == 'fro':
            graph_idx = self.graph_fro_DEAP
        elif graph == 'gen':
            graph_idx = self.graph_gen_DEAP
        elif graph == 'hem':
            graph_idx = self.graph_hem_DEAP
        elif graph == 'BL':
            graph_idx = self.original_order
        elif graph == 'TS':
            graph_idx = self.TS

        idx = []
        if graph in ['BL', 'TS']:
            for chan in graph_idx:
                idx.append(self.original_order.index(chan))
        else:
            num_chan_local_graph = []
            for i in range(len(graph_idx)):
                num_chan_local_graph.append(len(graph_idx[i]))
                for chan in graph_idx[i]:
                    idx.append(self.original_order.index(chan))

            # save the number of channels in local graph for building the LGG model in utils.py
            dataset = h5py.File(
                'num_chan_local_graph_{}.hdf'.format(graph), 'w')
            dataset['data'] = num_chan_local_graph
            dataset.close()
        return data[:, idx, :]

    def save(self, data, label, sub):
        """
        This function save the processed data into target folder
        Parameters
        ----------
        data: the processed data
        label: the corresponding label
        sub: the subject ID

        Returns
        -------
        None
        """
        save_path = os.getcwd()
        data_type = 'data_{}_{}_{}'.format(
            self.args.data_format, self.args.dataset, self.args.label_type)
        save_path = osp.join(save_path, data_type)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        else:
            pass
        name = 'sub' + str(sub) + '.hdf'
        save_path = osp.join(save_path, name)
        with h5py.File(save_path, 'w') as f:
            f['data'] = data
            f['label'] = label
        f.close()

    def split(self, data, label, segment_length=1, overlap=0, sampling_rate=256):
        """
        This function split one trial's data into shorter segments
        Parameters
        ----------
        data: (trial, f, channel, data)
        label: (trial,)
        segment_length: how long each segment is (e.g. 1s, 2s,...)
        overlap: overlap rate
        sampling_rate: sampling rate

        Returns
        -------
        data:(tiral, num_segment, f, channel, segment_legnth)
        label:(trial, num_segment,)
        """
        data_shape = data.shape
        step = int(segment_length * sampling_rate * (1 - overlap))
        data_segment = sampling_rate * segment_length
        data_split = []

        number_segment = int((data_shape[-1] - data_segment) // step)
        for i in range(number_segment + 1):
            data_split.append(
                data[:, :, :, (i * step):(i * step + data_segment)])
        data_split_array = np.stack(data_split, axis=1)
        label = np.stack([np.repeat(label[i], int(number_segment + 1))
                         for i in range(len(label))], axis=0)
        print("The data and label are split: Data shape:" + str(data_split_array.shape) + " Label:" + str(
            label.shape))
        data = data_split_array
        assert len(data) == len(label)
        return data, label
