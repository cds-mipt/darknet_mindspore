import os 
import datetime
from pprint import pprint
from mindspore.common import set_seed
from mindspore import context
from mindspore.train.callback import LossMonitor
from mindspore.train import Model
from mindspore.nn import Momentum
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, SummaryCollector

from src.config import get_config
from src.darknet import darknet53
from src.dataset import create_dataset
from src.cross_entropy_smooth import CrossEntropySmooth
from src.custom_callbacks import Evaluation_callback
from evaluation import test_net 

set_seed(1)

if __name__ == '__main__':

    #1. Configuring the Running Information
    config = get_config("default_config.yaml")
    context.set_context(mode=context.GRAPH_MODE if config.context_mode == "GRAPH_MODE" else context.PYNATIVE_MODE, 
                        device_target=config.device_target,
                        device_id=config.device_id)

    #config.outputs_dir = os.path.join(config.save_ckpt_path, datetime.datetime.now().strftime('%Y-%m-%d_time_%H_%M_%S'))

    #2. Processing Data
    train_dataset = create_dataset(dataset_path=config.dataset_path, 
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
    model = Model(net, loss, opt, metrics={'top_1_accuracy', 'top_5_accuracy'})

    config_ck = CheckpointConfig(save_checkpoint_steps=config.save_ckpt_step, keep_checkpoint_max=config.keep_checkpoint)
    ckpoint_cb = ModelCheckpoint(prefix="checkpoint_darknet53", directory=config.save_ckpt_path, config=config_ck) 
    summary_collector = SummaryCollector(summary_dir=config.summary_recorder_path, collect_freq=10)
    callbacks = [ckpoint_cb, LossMonitor(300), summary_collector]
    if config.eval_during_training:
        statistics = {"epoch_num": [], "top1_acc": [], "top2_acc": []}
        callbacks.append(Evaluation_callback(model=model, config=config, eval_per_epoch=config.eval_per_epoch, statistics=statistics))

    model.train(config.num_epoch, train_dataset, callbacks=callbacks, dataset_sink_mode=False)

    #6. Print statistics
    pprint(statistics)

