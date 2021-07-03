import logging


class Logger:

    def __init__(self):
        super(Logger, self).__init__()
        self.logger = self.get()

    def get(self):
        formatter = logging.Formatter(fmt='[%(asctime)s] <%(levelname)3s %(pathname)-10s:%(lineno)s> %(message)s', datefmt='%H:%M:%S')

        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger = logging.getLogger(self.__class__.__name__)
        if not hasattr(self, 'LOGLEVEL'):
            print('Class that inerit from Logger must inerit from Default config class')
            raise AttributeError

        logger.setLevel(self.LOGLEVEL)
        logger.addHandler(handler)
        return logger


