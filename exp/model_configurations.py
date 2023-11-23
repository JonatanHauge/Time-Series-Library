    


class ModelConfig:
    def __init__(self, task_name = 'long_term_forecast', is_training = True, model_id = 'test', model = 'Autoformer',
                 data = 'weather', root_path = r"C:\Users\johau\Desktop\Weel & Sandvig\ML\Time-Series-Library\dataset\weather\weather",
                 data_path = 'weather.csv', features = 'M', freq = 'h', checkpoints = './checkpoints/',
                 seq_len = 96, label_len = 48, pred_len = 96, seasonal_patterns = 'Monthly', inverse = False, mask_rate = 0.25,
                 anomaly_ratio = 0.25, top_k = 5, num_kernels = 6, enc_in=7, dec_in = 7, c_out=7, d_model=512, n_heads = 8,
                e_layers = 2, d_layers = 1, d_ff=2048, moving_avg=25, factor = 1, distil= True, dropout = 0.1, embed='timeF',
                activation = 'gelu', output_attention = False, num_workers = 10, itr = 1, train_epochs = 10, batch_size=32,
                patience = 3, learning_rate = 0.0001, des = 'test', loss = 'MSE', lradj = 'type1', use_amp = False,
                use_gpu = False, gpu = 0, use_multi_gpu = False, devices = '0,1,2,3', p_hidden_dims = [128,128],
                p_hidden_layers = 2, setting = 0, remove_cols = [None], scale = False, data_cut_low = 0, data_cut_high = 10000,
                target = 'All'):
        
        # basic config
        self.task_name = task_name
        self.is_training = is_training
        self.model_id = model_id
        self.model = model

        # data loader
        self.data = data
        self.root_path = root_path
        self.data_path = data_path
        self.remove_cols = remove_cols
        self.features = features
        self.freq = freq
        self.checkpoints = checkpoints
        self.scale = scale
        self.data_cut_low = data_cut_low
        self.data_cut_high = data_cut_high

        # forecasting task
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.seasonal_patterns = seasonal_patterns
        self.inverse = inverse
        self.target = target

        # inputation task
        self.mask_rate = mask_rate

        # anomaly detection task
        self.anomaly_ratio = anomaly_ratio

        # model define
        self.dec_in = dec_in
        self.n_heads = n_heads
        self.d_layers = d_layers
        self.moving_avg = moving_avg
        self.factor = factor
        self.distil = distil
        self.activation = activation
        self.dropout = dropout
        self.d_ff = d_ff
        self.enc_in = enc_in
        self.output_attention = output_attention
        self.dropout = dropout
        self.c_out = c_out
        self.e_layers = e_layers
        self.top_k = top_k
        self.d_model = d_model
        self.num_kernels = num_kernels
        self.embed = embed
        self.c_out = c_out


         # optimization
        self.num_workers = num_workers
        self.itr = itr
        self.train_epochs = train_epochs
        self.batch_size = batch_size
        self.patience = patience
        self.learning_rate = learning_rate
        self.des = des
        self.loss = loss
        self.lradj = lradj
        self.use_amp = use_amp
        
        # GPU
        self.use_gpu = use_gpu
        self.gpu = gpu
        self.use_multi_gpu = use_multi_gpu
        self.devices = devices

        # de-stationary projector params
        self.p_hidden_dims = p_hidden_dims
        self.p_hidden_layers = p_hidden_layers


        
        self.setting = 0
        if self.is_training:
            for ii in range(self.itr):
            # setting record of experiments
                setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
                    self.task_name,
                    self.model_id,
                    self.model,
                    self.data,
                    self.features,
                    self.seq_len,
                    self.label_len,
                    self.pred_len,
                    self.d_model,
                    self.n_heads,
                    self.e_layers,
                    self.d_layers,
                    self.d_ff,
                    self.factor,
                    self.embed,
                    self.distil,
                    self.des, ii)
            self.setting = setting


  