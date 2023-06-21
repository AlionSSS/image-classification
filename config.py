import os
import warnings


class DefaultConfig(object):
    visdom_env = "default"
    model = "SqueezeNet"
    data_path = "dogs-vs-cats-redux-kernels-edition"
    train_data_root = os.path.join(data_path, "train")
    test_data_root = os.path.join(data_path, "test")
    load_model_path = None

    use_gpu = True
    print_freq = 20

    debug_file = "/tmp/debug"
    result_file = "result.csv"

    max_epoch = 10
    batch_size = 128
    num_worker = 4
    lr = 0.1
    lr_decay = 0.95
    weight_decay = 0.0001

    def parse(self, **kwargs):
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn(f"Warning: opt has not attribute {k}")
            setattr(self, k, v)

    def print_attr(self):
        print("user config:")
        for k, v in self.__class__.__dict__.items():
            if not k.startswith("__"):
                print("\t" + k + " = " + str(getattr(self, k)))
