import utils

class BaseTrainer:
    def __init__(self, arg, torch_device, logger):
        self.torch_device = torch_device 
        
        self.model_type = arg.model

        self.epoch = arg.epoch
        self.start_epoch = 0

        self.batch_size = arg.batch_size
        
        self.save_path = arg.save_dir

        self.logger = logger
    
    def save(self):
        raise NotImplementedError("notimplemented save method")

    def load(self):
        raise NotImplementedError("notimplemented save method")

    def train(self):
        raise NotImplementedError("notimplemented save method")

    def valid(self):
        raise NotImplementedError("notimplemented valid method")

    def test(self):
        raise NotImplementedError("notimplemented test method")

    def inference(self):
        raise NotImplementedError("notimplemented interence method")
        
        
