#https://www.mindspore.cn/tutorial/en/0.3.0-alpha/advanced_use/customized_debugging_information.html
#https://www.mindspore.cn/docs/programming_guide/en/r1.3/evaluate_the_model_during_training.html?highlight=mindspore%20eval

from mindspore.train.callback import Callback
from evaluation import test_net

class Evaluation_callback(Callback):

    def __init__(self, model, config, eval_per_epoch, statistics):
        super().__init__()
        self.model = model
        self.config = config
        self.eval_per_epoch = eval_per_epoch
        self.statistics = statistics # {"epoch_num": [], "top1_acc": [], "top2_acc": []}

    def epoch_end(self, run_context):
        cb_param = run_context.original_args()
        cur_epoch = cb_param.cur_epoch_num
        if cur_epoch % self.eval_per_epoch == 0:
            print("============== Epoch end:{} ==============".format(cur_epoch))
            acc = test_net(self.config, model=self.model)
            self.statistics["epoch_num"].append(cur_epoch)
            self.statistics["top1_acc"].append(acc['top_1_accuracy'])
            self.statistics["top2_acc"].append(acc['top_5_accuracy'])
            print("==========================================")

