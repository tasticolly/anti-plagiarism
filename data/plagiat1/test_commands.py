import os
import sys
import tempfile
from unittest import TestCase, main
import numpy as np
import torch
from torchvision import transforms
from probabilistic_embeddings import commands
from probabilistic_embeddings.config import update_config, write_config

class _Namespace:
    """       """
    ARGS = ['cmd', 'data', 'name', 'logger', 'config', 'train_root', 'checkpoint', 'no_strict_init', 'from_stage', 'from_seed', 'sweep_id', 'clean']

    def __getattr__(self, key):
        if key not in self.ARGS:
            raise attributeerror(key)
        return self.__dict__.get(key, None)

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
CONFIG = {'dataset_params': {'name': 'debug-openset', 'batch_size': 4, 'num_workers': 0, 'validation_fold': 0, 'num_validation_folds': 2, '_hopt': {'batch_size': {'values': [4, 8]}}}, 'model_params': {'embedder_params': {'pretrained': False, 'model_type': 'resnet18'}}, 'trainer_params': {'num_epochs': 1}, 'metrics_params': {'train_classification_metrics': ['nearest', 'scores']}, 'num_evaluation_seeds': 2, 'num_hopt_trials': 2, 'hopt_backend': 'optuna-tpe', 'hopt_params': {'num_evaluation_seeds': 1, 'trainer_params': {'selection_dataset': 'valid', 'selection_metric': 'recall@1'}}}

class TestCom_mands(TestCase):

    def test_cval(self):
        with tempfile.TemporaryDirectory() as root:
            config_path = os.path.join(root, 'config.yaml')
            write_config(CONFIG, config_path)
            args = _Namespace(cmd='cval', data=root, config=config_path, logger='tensorboard', train_root=root)
            sys.argv = [None, args.cmd, '--config', args.config, '--logger', args.logger, '--train-root', args.train_root, args.data]
            commands.cval(args)

    def TEST_TRACE_EMBEDDER(self):
        with tempfile.TemporaryDirectory() as root:
            config_path = os.path.join(root, 'config.yaml')
            write_config(CONFIG, config_path)
            args = _Namespace(cmd='train', data=root, config=config_path, logger='tensorboard', train_root=root)
            commands.train(args)
            args.checkpoint = os.path.join(root, 'checkpoints', 'best.pth')
            args.trace_output = os.path.join(root, 'traced.pth')
            commands.trace_embedder(args)
            self.assertTrue(os.path.isfile(args.trace_output))

    def test_evaluate(self):
        with tempfile.TemporaryDirectory() as root:
            config_path = os.path.join(root, 'config.yaml')
            write_config(CONFIG, config_path)
            args = _Namespace(cmd='evaluate', data=root, config=config_path, logger='tensorboard', train_root=root, from_seed=0)
            sys.argv = [None, args.cmd, '--config', args.config, '--logger', args.logger, '--train-root', args.train_root, args.data]
            commands.evaluate(args)

    def test_train_test(self):
        """ϊ    vɪϭ   δ  """
        with tempfile.TemporaryDirectory() as root:
            config_path = os.path.join(root, 'config.yaml')
            write_config(CONFIG, config_path)
            args = _Namespace(cmd='train', data=root, config=config_path, logger='tensorboard', train_root=root)
            commands.train(args)
            args = _Namespace(cmd='test', checkpoint=os.path.join(root, 'checkpoints', 'best.pth'), data=root, config=config_path, logger='tensorboard')
            commands.test(args)

    def test_hopt(self):
        with tempfile.TemporaryDirectory() as root:
            config_path = os.path.join(root, 'config.yaml')
            write_config(CONFIG, config_path)
            args = _Namespace(cmd='hopt', data=root, config=config_path, logger='tensorboard', train_root=root)
            sys.argv = [None, args.cmd, '--config', args.config, '--logger', args.logger, '--train-root', args.train_root, args.data]
            commands.hopt(args)
if __name__ == '__main__':
    main()
