import pandas as pd
import numpy as np
import torch
import glob


class Dataset(object):
    def __init__(self, path, sep=',', session_key='SessionID', item_key='ItemID', time_key='Time',
                 n_sample=-1, itemmap=None, usermap=None, itemstamp=None, time_sort=False):
        # Read csv

        self.df = self.read_csv_gz(path, sep, session_key, item_key, time_key)
        self.session_key = session_key
        self.item_key = item_key
        self.time_key = time_key
        self.time_sort = time_sort
        if n_sample > 0:
            self.df = self.df[:n_sample]

        # Add colummn item index to data
        self.add_item_indices(itemmap=itemmap)
        self.add_user_indices(usermap=usermap)
        """
        Sort the df by time, and then by session ID. That is, df is sorted by session ID and
        clicks within a session are next to each other, where the clicks within a session are time-ordered.
        """
        self.df.sort_values(['user_idx', time_key], inplace=True)
        self.df.rename(
            columns={
                "SessionID": "SessionID_raw",
                "user_idx": "SessionID"},
            inplace=True)
        self.df = self.df[['SessionID', 'Time',
                           'ItemID', 'item_idx', 'SessionID_raw']]
        self.df.drop('SessionID_raw', axis=1, inplace=True)
        self.df = self.df.reset_index(drop=True)
        self.user_action_count = self.user_action_count_filter_func()
        self.df = pd.merge(
            self.df,
            self.user_action_count,
            on='SessionID',
            how='inner')
        self.df = self.df.groupby('SessionID', as_index=False).apply(lambda x: x[-50:]).reset_index(drop=True)
        self.click_offsets = self.get_click_offset()
        self.session_idx_arr = self.order_session_idx()


        self.sss = 0
    def read_csv_gz(self, path, sep, session_key, item_key, time_key):
        paths = glob.glob(path)
        df_array = []
        for path in paths:
            df_temp = pd.read_csv(
                path,
                sep=sep,
                dtype={
                    'uuid': str,
                    'item_id': str,
                    'timestamp': int},
                compression='gzip')
            df_temp.columns=[session_key, time_key, item_key]
            df_array.append(df_temp)
        df = pd.concat(df_array)
        df = df.reset_index(drop=True)
        return df

    def add_item_indices(self, itemmap=None):
        """
        Add item index column named "item_idx" to the df
        Args:
            itemmap (pd.DataFrame): mapping between the item Ids and indices
        """
        if itemmap is None:
            item_ids = self.df[self.item_key].unique()  # type is numpy.ndarray
            item2idx = pd.Series(data=np.arange(len(item_ids)),
                                 index=item_ids)
            # Build itemmap is a DataFrame that have 2 columns (self.item_key,
            # 'item_idx)
            itemmap = pd.DataFrame({self.item_key: item_ids,
                                    'item_idx': item2idx[item_ids].values})
        self.itemmap = itemmap
        self.df = pd.merge(
            self.df,
            self.itemmap,
            on=self.item_key,
            how='inner')

    def add_user_indices(self, usermap=None):
        """
        replace user index column named "user_idx" to the df
        Args:
            usermap (pd.DataFrame): mapping between the SessionID and indices
        """
        if usermap is None:
            # type is numpy.ndarray
            user_ids = self.df[self.session_key].unique()
            user2idx = pd.Series(data=np.arange(len(user_ids)),
                                 index=user_ids)
            # Build usermap is a DataFrame that have 2 columns (self.item_key,
            # 'item_idx)
            usermap = pd.DataFrame({self.session_key: user_ids,
                                    'user_idx': user2idx[user_ids].values})
        self.usermap = usermap
        self.df = pd.merge(
            self.df,
            self.usermap,
            on=self.session_key,
            how='inner')

    def get_click_offset(self):
        """
        self.df[self.session_key] return a set of session_key
        self.df[self.session_key].nunique() return the size of session_key set (int)
        self.df.groupby(self.session_key).size() return the size of each session_id
        self.df.groupby(self.session_key).size().cumsum() retunn cumulative sum
        """
        offsets = np.zeros(
            self.df[self.session_key].nunique() + 1, dtype=np.int32)
        offsets[1:] = self.df.groupby(self.session_key).size().cumsum()
        return offsets

    def order_session_idx(self):
        if self.time_sort:
            sessions_start_time = self.df.groupby(self.session_key)[
                self.time_key].min().values
            session_idx_arr = np.argsort(sessions_start_time)
        else:
            session_idx_arr = np.arange(self.df[self.session_key].nunique())
        return session_idx_arr

    def user_action_count_filter_func(self):
        """
        ?????????????????????>2???????????????????????????????????????df???merge????????????????????????????????????
        """
        user_action_count = self.df.groupby('SessionID', as_index=False).count()[['SessionID', 'Time']]
        user_action_count_filter = user_action_count[user_action_count['Time']>1][['SessionID']]
        return  user_action_count_filter


    @property
    def items(self):
        return self.itemmap[self.item_key].unique()


class DataLoader():
    def __init__(self, dataset, batch_size=50, if_predict=False):
        """
        A class for creating session-parallel mini-batches.

        Args:
             dataset (SessionDataset): the session dataset to generate the batches from
             batch_size (int): size of the batch
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.if_predict = if_predict

    def __iter__(self):
        """ Returns the iterator for producing session-parallel training mini-batches.

        Yields:
            input (B,): torch.FloatTensor. Item indices that will be encoded as one-hot vectors later.
            target (B,): a Variable that stores the target item indices
            masks: Numpy array indicating the positions of the sessions to be terminated
        """
        # initializations
        df = self.dataset.df
        click_offsets = self.dataset.click_offsets
        session_idx_arr = self.dataset.session_idx_arr

        iters = np.arange(self.batch_size)
        maxiter = iters.max()
        start = click_offsets[session_idx_arr[iters]]
        end = click_offsets[session_idx_arr[iters] + 1]
        mask = []  # indicator for the sessions to be terminated
        finished = False

        # if self.if_predict:
        #     mask = np.arange(len(iters))[(end - start) == 1]

        if self.if_predict:
            while not finished:
                minlen = (end - start).min()
                # Item indices(for embedding) for clicks where the first sessions
                # start
                # idx_target = df.item_idx.values[start]
                for i in range(minlen):
                    # idx_input = idx_target
                    # idx_target = df.item_idx.values[start + i + 1]
                    now = start+i
                    idx_input = df.item_idx.values[now]
                    input = torch.LongTensor(idx_input)
                    mask = np.arange(len(iters))[(end - now) == 1]
                    user_id = df.SessionID.values[now]
                    yield input, mask, user_id

                start = now + 1
                for idx in mask:
                    maxiter += 1
                    if maxiter >= len(click_offsets) - 1:
                        finished = True
                        break
                    iters[idx] = maxiter
                    start[idx] = click_offsets[session_idx_arr[maxiter]]
                    end[idx] = click_offsets[session_idx_arr[maxiter] + 1]

        else:
            while not finished:
                minlen = (end - start).min()
                # Item indices(for embedding) for clicks where the first sessions
                # start
                idx_target = df.item_idx.values[start]
                for i in range(minlen - 1):
                    # Build inputs & targets
                    idx_input = idx_target
                    idx_target = df.item_idx.values[start + i + 1]
                    input = torch.LongTensor(idx_input)
                    target = torch.LongTensor(idx_target)
                    yield input, target, mask

                # click indices where a particular session meets second-to-last
                # element
                start = start + (minlen - 1)
                # see if how many sessions should terminate
                mask = np.arange(len(iters))[(end - start) <= 1]
                for idx in mask:
                    maxiter += 1
                    if maxiter >= len(click_offsets) - 1:
                        finished = True
                        break
                    # update the next starting/ending point
                    iters[idx] = maxiter
                    start[idx] = click_offsets[session_idx_arr[maxiter]]
                    end[idx] = click_offsets[session_idx_arr[maxiter] + 1]
