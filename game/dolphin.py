
from subprocess import Popen
from time import sleep
import signal
import configparser
import pprint
import filelock

from codes.boot2match import generate_match_code
from codes import codes
from config.loader import Default


class DolphinInstance(Default):

    def __init__(self, exe_path , iso_path, test, ID, config_path='dolphin/User/GameSettings/GALE01.ini'):
        super(DolphinInstance, self).__init__()

        self.exe_path = exe_path
        self.iso_path = iso_path
        self.id = ID
        self.test = test
        self.proc = None
        self.config_path = config_path
        self.lock = filelock.FileLock(config_path+'.lock')

    def run(self, *players):
        enabled = codes.demo if self.test else codes.training
        game_config = codes.GALE01_ini_template.format(match_code=generate_match_code(*players, stage=self.stage),

                                                 enabled='\n'.join(enabled))
        cmd = r'%s --exec=%s --id=%d' \
              % (self.exe_path, self.iso_path, self.id)
        if not self.test:
            cmd += ' --platform=headless'
        with self.lock :
            with open(self.config_path, 'w') as gconfigfile:
                gconfigfile.write(game_config)
            self.proc = Popen(cmd.split(), start_new_session=True)
            sleep(0.2)

    def close(self):
        try:
            self.proc.send_signal(signal.SIGUSR1)
            print('Sent signal to Dolphin process ')

            self.proc.wait()

        except Exception as e:
            pass