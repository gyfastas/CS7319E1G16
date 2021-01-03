#!/usr/bin/env
# -*- coding: utf-8 -*-

'''
Copyright (c) 2019 Charles
'''
import os
import json
import time
import logging
import numpy as np
from tqdm import tqdm


__all__ = ['Logger']


class TqdmHandler(logging.StreamHandler):
    def __init__(self):
        logging.StreamHandler.__init__(self)

    def emit(self, record):
        msg = self.format(record)
        tqdm.write(msg)


class Logger(object):
    def __init__(self, args, filename=None, mode=None):
        self.args = args
        self.workdir = args.workdir
        self.timestamp = args.timestamp
        if filename is None:
            filename = '{}.log.txt'.format(self.timestamp)
        self.filename = os.path.join(self.workdir, filename)

        # Redirect stdout to tqdm.write
        logging.root.handlers = []
        
        if mode is None:
            mode = 'a' if os.path.exists(self.filename) else 'w'
        logging.basicConfig(
            format='%(asctime)s - %(levelname)s -   %(message)s',
            datefmt='%m/%d/%Y %H:%M:%S',
            level=logging.INFO,
            handlers=[
                TqdmHandler(),
                logging.FileHandler(
                    self.filename, mode=mode)
            ]
        )
        self.logger = logging.getLogger()
        
    def init_info(self, save_config=True):
        self.info(
            '------------------- * {} * -------------------'.format(self.timestamp))
        self.info(
            'Working directory: {}'.format(self.workdir))
        self.info('Experiment log saves to {}'.format(self.filename))
        if save_config:
            self.info('Experiment configuration saves to {}:'.format(
                os.path.join(self.workdir, 'config.json')))
            with open(os.path.join(self.workdir, 'config.json'), 'w') as f:
                json.dump(self.args.__dict__, f, indent=4)

    def info(self, *args, **kwargs):
        self.logger.info(*args, **kwargs)

    def warning(self, *args, **kwargs):
        self.logger.warning(*args, **kwargs)