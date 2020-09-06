import time
import tqdm
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, List, Union, Callable, Optional

from tianshou.data import Collector
from tianshou.policy import BasePolicy
from tianshou.utils import tqdm_config, MovAvg
from tianshou.trainer import test_episode, gather_info


def onpolicy_trainer(
        policy: BasePolicy,
        train_collector: Collector,
        max_epoch: int,
        step_per_epoch: int,
        collect_per_step: int,
        repeat_per_collect: int,
        batch_size: int,
        train_fn: Optional[Callable[[int], None]] = None,
        log_interval: int = 1,
        verbose: bool = True,
        test_in_train: bool = True,
        writer: Optional[SummaryWriter] = None,
):
    """A wrapper for on-policy trainer procedure. The ``step`` in trainer means
    a policy network update.

    :param policy: an instance of the :class:`~tianshou.policy.BasePolicy`
        class.
    :param train_collector: the collector used for training.
    :type train_collector: :class:`~tianshou.data.Collector`
    :param test_collector: the collector used for testing.
    :type test_collector: :class:`~tianshou.data.Collector`
    :param int max_epoch: the maximum of epochs for training. The training
        process might be finished before reaching the ``max_epoch``.
    :param int step_per_epoch: the number of step for updating policy network
        in one epoch.
    :param int collect_per_step: the number of episodes the collector would
        collect before the network update. In other words, collect some
        episodes and do one policy network update.
    :param int repeat_per_collect: the number of repeat time for policy
        learning, for example, set it to 2 means the policy needs to learn each
        given batch data twice.
    :param int batch_size: the batch size of sample data, which is going to
        feed in the policy network.
    :param function train_fn: a function receives the current number of epoch
        index and performs some operations at the beginning of training in this
        epoch.
    :param torch.utils.tensorboard.SummaryWriter writer: a TensorBoard
        SummaryWriter.
    :param int log_interval: the log interval of the writer.
    :param bool verbose: whether to print the information.
    :param bool test_in_train: whether to test in the training phase.

    :return: See :func:`~tianshou.trainer.gather_info`.
    """
    global_step = 0
    best_epoch, best_reward = -1, -1
    stat = {}
    start_time = time.time()
    test_in_train = test_in_train and train_collector.policy == policy
    for epoch in range(1, 1 + max_epoch):
        # train
        policy.train()
        if train_fn:
            train_fn(epoch)
        with tqdm.tqdm(total=step_per_epoch, desc=f'Epoch #{epoch}',
                       **tqdm_config) as t:
            while t.n < t.total:
                result = train_collector.collect(n_episode=collect_per_step)
                data = {}
                losses = policy.update(
                    0, train_collector.buffer, batch_size, repeat_per_collect)
                train_collector.reset_buffer()
                step = 1
                for k in losses.keys():
                    if isinstance(losses[k], list):
                        step = max(step, len(losses[k]))
                global_step += step
                for k in result.keys():
                    data[k] = f'{result[k]:.2f}'
                    if writer and global_step % log_interval == 0:
                        writer.add_scalar(
                            k, result[k], global_step=global_step)
                for k in losses.keys():
                    if stat.get(k) is None:
                        stat[k] = MovAvg()
                    stat[k].add(losses[k])
                    data[k] = f'{stat[k].get():.6f}'
                    if writer and global_step % log_interval == 0:
                        writer.add_scalar(
                            k, stat[k].get(), global_step=global_step)
                t.update(step)
                t.set_postfix(**data)
            if t.n <= t.total:
                t.update()
    return global_step
