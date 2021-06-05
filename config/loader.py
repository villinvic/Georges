import configparser
import inspect
import os

from numpy import float32, int32
import pandas


class ConfigLoader:
    """
    Global Config Loader helper class
    """
    parser = configparser.ConfigParser()

    @staticmethod
    def load(config_file):
        ConfigLoader.parser.read(config_file)
        return ConfigLoader.parser


class Default:
    """
    Base Class for every class that need config data
    Package P,  class C(Default) -> loads config in config/p.ini at class C index
    """

    types = {
        'float': float32,
        'int': int32,
        'str': str,
    }

    def __init__(self):
        configfile = 'config/' + inspect.getmodule(self).__file__.split('/')[-2].lower() + '.ini'

        self.config = ConfigLoader.load(configfile)

        for k,v in self.config[self.__class__.__name__].items():
            t, value = v.split()
            if t == 'csv':
                with open('config/'+value+'.csv', newline='') as file:
                    data = pandas.read_csv(file, delimiter=',')
                setattr(self, k, data.itertuples(index=False))

            else:
                setattr(self, k, self.types[t](value))

        if 'Global' in self.config:
            for k, v in self.config['Global'].items():
                t, value = v.split()
                setattr(self, k.upper(), self.types[t](value))

        super(Default, self).__init__()


