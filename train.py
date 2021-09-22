import os 
import datetime
from mindspore.common import set_seed
from mindspore import context
from mindspore.nn.metrics import Accuracy
from mindspore.train.callback import LossMonitor
from mindspore.train import Model
from mindspore.nn import Momentum
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig

from src.config import get_config
from src.darknet import darknet53
from src.dataset import create_dataset
from src.cross_entropy_smooth import CrossEntropySmooth

set_seed(1)

if __name__ == '__main__':

    #1. Configuring the Running Information
    config = get_config("default_config.yaml")
    context.set_context(mode=context.GRAPH_MODE if config.context_mode == "GRAPH_MODE" else context.PYNATIVE_MODE, 
                        device_target=config.device_target,
                        device_id=config.device_id)

    #config.outputs_dir = os.path.join(config.save_ckpt_path, datetime.datetime.now().strftime('%Y-%m-%d_time_%H_%M_%S'))

    #2. Processing Data
    dataset = create_dataset(dataset_path=config.dataset_path, 
                             do_train=True, 
                             repeat_num=config.repeat_num,
                             batch_size=config.batch_size, 
                             target=config.device_target, 
                             distribute=config.run_distribute,
                             num_parallel_workers=config.num_parallel_workers)

    #3. Defining the Network
    net = darknet53()

    #4. Defining the Loss Function and Optimizer
    loss = CrossEntropySmooth(sparse=True, 
                              reduction="mean",
                              smooth_factor=0.0, 
                              num_classes=1000)

    opt = Momentum(net.trainable_params(), 0.01, 0.9)

    #5. Training the Network
    config_ck = CheckpointConfig(save_checkpoint_steps=config.save_ckpt_step, keep_checkpoint_max=config.keep_checkpoint)
    ckpoint_cb = ModelCheckpoint(prefix="checkpoint_darknet53", config=config_ck) 

    model = Model(net, loss, opt, metrics={"Accuracy": Accuracy()})
    model.train(config.num_epoch, dataset, callbacks=[ckpoint_cb, LossMonitor()], dataset_sink_mode=False)
    # import numpy as np
    # import mindspore
    # test = mindspore.Tensor(np.random.random_sample((1, 3, 256, 256)), mindspore.float32)
    # print(test.shape)
    # out = net(test)
    # print(out.shape)

