from sklearn.preprocessing import StandardScaler
import pandas as pd
from torch.utils.data import Dataset
from utils.timefeatures import time_features

class Dataset_WS(Dataset):
    def __init__(self, path, flag='train', scale=True, freq = 't', size = [60,10,10], remove_cols = [None], 
                 data_cut_low = 0, data_cut_high = 10000):
        # size [seq_len, label_len, pred_len]
        # info
        self.data_cut_low = data_cut_low
        self.data_cut_high = data_cut_high
        self.freq = freq

        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.scale = scale
        self.path = path
        self.remove_cols = remove_cols
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(self.path)
        df_raw = df_raw.iloc[self.data_cut_low : self.data_cut_high]

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        df_data = df_raw.drop(self.remove_cols, axis = 1)
        

        if self.scale:
            train_data = df_data.iloc[border1s[0]:border2s[0], 1:]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.iloc[:,1:].values)
        else:
            data = df_data.iloc[:,1:].values

        df_stamp = df_raw.inputtimestamp[border1:border2].to_frame()
        df_stamp['date'] = pd.to_datetime(df_stamp.inputtimestamp)

        data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
        data_stamp = data_stamp.transpose(1, 0)
        #df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
        #df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
        #df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
        #df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
        #data_stamp = df_stamp.drop(['date', 'inputtimestamp'], 1).values


        self.data_stamp = data_stamp

        self.data_x = data[border1:border2].astype(float)
        self.data_y = data[border1:border2].astype(float)

    def __getitem__(self, index):
        s_begin = index                   # Vi går kun 1 skridt frem for hver kald, sæt s_begin til at være index*skridtlængde hvis dette skal ændres.
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)