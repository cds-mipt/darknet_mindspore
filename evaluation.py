from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.train import Model
from mindspore import context

from src.config import get_config
from src.darknet import darknet53
from src.dataset import create_dataset
from src.cross_entropy_smooth import CrossEntropySmooth


def test_net(config, network=None, model=None, from_checkpoint=False):
    dataset_eval = create_dataset(dataset_path=config.eval_path, 
                                  do_train=False, 
                                  repeat_num=1,
                                  batch_size=config.batch_size, 
                                  target=config.device_target, 
                                  distribute=config.run_distribute,
                                  num_parallel_workers=config.num_parallel_workers)

    if from_checkpoint:
        assert network is None
        param_dict = load_checkpoint(config.checkpoint_name)
        load_param_into_net(network, param_dict)

    if not model:
        assert network is None

        loss = CrossEntropySmooth(sparse=True, 
                              reduction="mean",
                              smooth_factor=0.0, 
                              num_classes=1000)
        model = Model(network, loss, metrics={'top_1_accuracy', 'top_5_accuracy'})

    print("============== Starting Testing ==============")
    acc = model.eval(dataset_eval, dataset_sink_mode=False)
    print("============== Accuracy:{} ==============".format(acc))
    return acc


if __name__ == '__main__':

    config = get_config("default_config.yaml")
    context.set_context(mode=context.GRAPH_MODE if config.context_mode == "GRAPH_MODE" else context.PYNATIVE_MODE, 
                        device_target=config.device_target,
                        device_id=config.device_id)

    test_net(config, network=darknet53(), from_checkpoint=True)