import numpy as np
import datetime
import os
import csv
import h5py
import copy
import os.path as osp
from train_model import *
from utils import Averager, ensure_path
from sklearn.model_selection import KFold
import pickle

ROOT = os.getcwd()


class CrossValidation:
    def __init__(self, args):
        self.args = args
        self.data = None
        self.label = None
        self.model = None
        # Log the results per subject
        result_path = osp.join(args.save_path, 'result')
        ensure_path(result_path)
        self.text_file = osp.join(result_path,
                                  "results_{}.txt".format(args.dataset))
        file = open(self.text_file, 'a')
        file.write("\n" + str(datetime.datetime.now()) + "\nTrain:Parameter setting for " + str(args.model) + ' on ' + str(args.dataset) +
                   # Remove number_class parameter since it's not needed for regression
                   "\n1)random_seed:" + str(args.random_seed) +
                   "\n2)learning_rate:" + str(args.learning_rate) +
                   "\n3)pool:" + str(args.pool) +
                   "\n4)num_epochs:" + str(args.max_epoch) +
                   "\n5)batch_size:" + str(args.batch_size) +
                   "\n6)dropout:" + str(args.dropout) +
                   "\n7)hidden_node:" + str(args.hidden) +
                   "\n8)input_shape:" + str(args.input_shape) +
                   # Changed class to target
                   "\n9)target:" + str(args.label_type) +
                   "\n10)T:" + str(args.T) +
                   "\n11)graph-type:" + str(args.graph_type) + '\n')
        file.close()

    def load_per_subject(self, sub):
        """
        load data for sub
        :param sub: which subject's data to load
        :return: data and label
        """
        save_path = os.getcwd()
        data_type = 'data_{}_{}_{}'.format(
            self.args.data_format, self.args.dataset, self.args.label_type)
        sub_code = 'sub' + str(sub) + '.hdf'
        path = osp.join(save_path, data_type, sub_code)
        dataset = h5py.File(path, 'r')
        data = np.array(dataset['data'])
        label = np.array(dataset['label'])
        print('>>> Data:{} Label:{}'.format(data.shape, label.shape))
        return data, label

    def prepare_data(self, idx_train, idx_test, data, label):
        """
        1. get training and testing data according to the index
        2. numpy.array-->torch.tensor
        :param idx_train: index of training data
        :param idx_test: index of testing data
        :param data: (segments, 1, channel, data)
        :param label: (segments,)
        :return: data and label
        """
        data_train = data[idx_train]
        label_train = label[idx_train]
        data_test = data[idx_test]
        label_test = label[idx_test]

        if self.args.dataset == 'Att' or self.args.dataset == 'DEAP':
            """
            For DEAP we want to do trial-wise 10-fold, so the idx_train/idx_test is for
            trials.
            data: (trial, segment, 1, chan, datapoint)
            To use the normalization function, we should change the dimension from
            (trial, segment, 1, chan, datapoint) to (trial*segments, 1, chan, datapoint)
            """
            data_train = np.concatenate(data_train, axis=0)
            label_train = np.concatenate(label_train, axis=0)
            if len(data_test.shape) > 4:
                """
                When leave one trial out is conducted, the test data will be (segments, 1, chan, datapoint), hence,
                no need to concatenate the first dimension to get trial*segments
                """
                data_test = np.concatenate(data_test, axis=0)
                label_test = np.concatenate(label_test, axis=0)

        data_train, data_test = self.normalize(
            train=data_train, test=data_test)
        # Prepare the data format for training the model using PyTorch
        data_train = torch.from_numpy(data_train).float()
        label_train = torch.from_numpy(label_train).float()  # Changed to float

        data_test = torch.from_numpy(data_test).float()
        label_test = torch.from_numpy(label_test).float()  # Changed to float
        return data_train, label_train, data_test, label_test

    def normalize(self, train, test):
        """
        this function do standard normalization for EEG channel by channel
        :param train: training data (sample, 1, chan, datapoint)
        :param test: testing data (sample, 1, chan, datapoint)
        :return: normalized training and testing data
        """
        # data: sample x 1 x channel x data
        for channel in range(train.shape[2]):
            mean = np.mean(train[:, :, channel, :])
            std = np.std(train[:, :, channel, :])
            train[:, :, channel, :] = (train[:, :, channel, :] - mean) / std
            test[:, :, channel, :] = (test[:, :, channel, :] - mean) / std
        return train, test

    def split_balance_class(self, data, label, train_rate, random):
        """
        Get the validation set using the same percentage of the two classe samples
        :param data: training data (segment, 1, channel, data)
        :param label: (segments,)
        :param train_rate: the percentage of trianing data
        :param random: bool, whether to shuffle the training data before get the validation data
        :return: data_trian, label_train, and data_val, label_val
        """
        # Data dimension: segment x 1 x channel x data
        # Label dimension: segment x 1
        np.random.seed(0)
        # data : segments x 1 x channel x data
        # label : segments

        index_0 = np.where(label == 0)[0]
        index_1 = np.where(label == 1)[0]

        # for class 0
        index_random_0 = copy.deepcopy(index_0)

        # for class 1
        index_random_1 = copy.deepcopy(index_1)

        if random == True:
            np.random.shuffle(index_random_0)
            np.random.shuffle(index_random_1)

        index_train = np.concatenate((index_random_0[:int(len(index_random_0) * train_rate)],
                                      index_random_1[:int(len(index_random_1) * train_rate)]),
                                     axis=0)
        index_val = np.concatenate((index_random_0[int(len(index_random_0) * train_rate):],
                                    index_random_1[int(len(index_random_1) * train_rate):]),
                                   axis=0)

        # get validation
        val = data[index_val]
        val_label = label[index_val]

        train = data[index_train]
        train_label = label[index_train]

        return train, train_label, val, val_label

    def n_fold_CV(self, subject=[0], fold=10, shuffle=True):
        """
        this function achieves n-fold cross-validation
        :param subject: how many subject to load
        :param fold: how many fold
        """

        mse_total = []  # Mean squared error
        mae_total = []  # Mean absolute error
        r2_total = []   # R-squared score


        for sub in subject:
            data, label = self.load_per_subject(sub)
            mse_val = Averager()
            mae_val = Averager()
            preds, acts = [], []
            kf = KFold(n_splits=fold, shuffle=shuffle)
            for idx_fold, (idx_train, idx_test) in enumerate(kf.split(data)):
                print('Outer loop: {}-fold-CV Fold:{}'.format(fold, idx_fold))
                data_train, label_train, data_test, label_test = self.prepare_data(
                    idx_train=idx_train, idx_test=idx_test, data=data, label=label)

                if self.args.reproduce:
                    mse_test, pred, act = test(args=self.args, data=data_test, label=label_test,
                                               reproduce=self.args.reproduce,
                                               subject=sub, fold=idx_fold)
                    mse_val_curr = 0
                    mae_val_curr = 0
                else:
                    mse_val_curr, mae_val_curr = self.first_stage(data=data_train, label=label_train,
                                                                  subject=sub, fold=idx_fold)

                    combine_train(args=self.args,
                                  data=data_train, label=label_train,
                                  subject=sub, fold=idx_fold)

                    mse_test, pred, act = test(args=self.args, data=data_test, label=label_test,
                                               reproduce=self.args.reproduce,
                                               subject=sub, fold=idx_fold)

                mse_val.add(mse_val_curr)
                mae_val.add(mae_val_curr)
                preds.extend(pred)
                acts.extend(act)

            mse, mae, r2 = get_metrics(y_pred=preds, y_true=acts)
            mse_total.append(mse)
            mae_total.append(mae)
            r2_total.append(r2)

            result = 'MSE:{:.4f}, MAE:{:.4f}, R2:{:.4f}'.format(mse, mae, r2)
            self.log2txt(result)

        # prepare final report
        final_mse = np.mean(mse_total)
        final_mae = np.mean(mae_total)
        final_r2 = np.mean(r2_total)

        print('Final: mean MSE:{:.4f}'.format(final_mse))
        print('Final: mean MAE:{:.4f}'.format(final_mae))
        print('Final: mean R2:{:.4f}'.format(final_r2))
        results = 'Final metrics: MSE={:.4f} MAE={:.4f} R2={:.4f}'.format(
            final_mse, final_mae, final_r2)
        self.log2txt(results)

    def first_stage(self, data, label, subject, fold):
        """
        this function achieves n-fold-CV to:
            1. select hyper-parameters on training data
            2. get the model for evaluation on testing data
        :param data: (segments, 1, channel, data)
        :param label: (segments,)
        :param subject: which subject the data belongs to
        :param fold: which fold the data belongs to
        :return: mean validation accuracy
        """
        # use n-fold-CV to select hyper-parameters on training data
        # save the best performance model and the corresponding acc for the second stage
        # data: trial x 1 x channel x time
        kf = KFold(n_splits=3, shuffle=True)
        val_loss_avg = Averager()  # 跟踪验证损失
        mae_avg = Averager()       # 跟踪平均绝对误差
        val_losses = []
        min_loss = float('inf')    # 跟踪最小验证损失
        for i, (idx_train, idx_val) in enumerate(kf.split(data)):
            print('Inner 3-fold-CV Fold:{}'.format(i))
            data_train, label_train = data[idx_train], label[idx_train]
            data_val, label_val = data[idx_val], label[idx_val]
            loss_val, mae_val = train(args=self.args,
                                 data_train=data_train,
                                 label_train=label_train,
                                 data_val=data_val,label_val=label_val,
                                 subject=subject,
                                 fold=fold)
            val_loss_avg.add(loss_val)
            mae_avg.add(mae_val)
            val_losses.append(loss_val)
            if loss_val <= min_loss:
                min_loss = loss_val
                # 重命名当前最佳模型
                old_name = osp.join(self.args.save_path, 'candidate.pth')
                new_name = osp.join(self.args.save_path, 'min-loss.pth')
                if os.path.exists(new_name):
                    os.remove(new_name)
                os.rename(old_name, new_name)
                print('New best model saved, with validation loss:{:.4f}'.format(loss_val))

        mean_loss = val_loss_avg.item()
        mean_mae = mae_avg.item()
        return mean_loss, mean_mae


    def log2txt(self, content):
        """
        this function log the content to results.txt
        :param content: string, the content to log
        """
        file = open(self.text_file, 'a')
        file.write(str(content) + '\n')
        file.close()
