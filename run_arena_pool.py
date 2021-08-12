from subprocess import Popen
from time import sleep
import signal
import sys
import fire
import os
import getpass

def run_many(n_arenas=1,
             hub_ip=None,
             restart_freq=60*60,
             ssh=False):

    if ssh:
        psw = getpass.getpass("Server Password: ")
    else:
        psw = '""'

    cmd = "python3 training/arena.py " \
          "--ID={ID} " \
          "--hub_ip={hub_ip} " \
          "--ssh={ssh}"

    if hub_ip is None:
        hub_ip = '127.0.0.1'


    procs = [None] * n_arenas
    
    def start():
        for ID in range(n_arenas):
            procs[ID] = Popen(cmd.format(ID=ID, hub_ip=hub_ip, ssh=psw).split(),
                              env=dict(os.environ, PYTHONPATH= os.getcwd()))


    def close():
        for ID in range(n_arenas):
            procs[ID].send_signal(signal.SIGINT)
        sleep(3)

    secs = 0
    try:
        start()
        while True:
            sleep(1)
            secs += 1
            if not secs % restart_freq:
                close()
                start()

    except KeyboardInterrupt:
        pass

    close()

if __name__ == '__main__':
    sys.exit(fire.Fire(run_many))

