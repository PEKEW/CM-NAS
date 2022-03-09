class Args:
    def __init__(self):
        self.degree = 4
        self.edge_num = 10
        self.data_path = 'data'
        self.save_path = 'saved_model'
        self.epochs = 30
        self.warm_up = 5
        self.end_epoch = 100
        self.train_portion = 0.5
        self.begin_op_nums = self.__get_op_nums()
        self.end_op_nums = 1
        self.end_op_groups = 4
        self.batch_size = 64
        self.learning_rate = 0.025
        self.min_learning_rate = 0.001
        self.momentum = 0.9
        self.weight_decay = 3e-4
        self.cutout = True
        self.cutout_length = 16
        self.grad_clip = 5
        self.drop = 0.3
        self.unrolled = False
        self.arch_learning_rate = 3e-4
        self.arch_weight_decay = 1e-3
        self.init_channel_nums = 16
        self.layers = 8
        self.seed = 2
        self.gpu_id = 0
        self.classes = 10
        self.report_frequent = 64
        self.mix_lambda = 1.0
        self.decay_rate = 0.1
        self.explore_lambda = 1e-3 * 1.4
        self.alpha_weight = [0.0, 1.0]
        self.path = 'log_/'

    def print(self):
        print(self.__dict__)


    @staticmethod
    def __get_op_nums():
        from genotypes import PRIMITIVES
        return len(PRIMITIVES)
