import melee
from codes import codes
import configparser

def main():
    console = melee.Console(path='..')

codes = "\n".join([codes.SPEEDHACK_NORENDER, codes.DMA, codes.BOOT2MATCH, codes.SKIP_MEMCARD])

enabled = """
$Required: Skip Memcard Prompt
$Required: Speedhack no render
$Required: Boot to match
$Required: Setup match
$Required: DMA Read Before Poll
        """


if __name__ == '__main__':
    console = melee.Console(path='..')
    controller = melee.Controller(console=console, port=2)

    config_path = console.dolphin_home_path + 'Config/Dolphin.ini'
    config = configparser.ConfigParser()
    config.read(config_path)
    config['Core']['GFXBackend'] = 'Null'
    config['Core']['slotb'] = '10'

    with open(config_path, 'w') as configfile:
        config.write(configfile)

    with open(console.dolphin_home_path + 'GameSettings/GALE01.ini', 'r') as f:
        lines = f.readlines()
    print(lines)
    new = []
    for l in lines:
        new.append(l)
        if '[Gecko]' in l:
            new.append(codes)
        elif '[Gecko_Enabled]' in l:
            new.append(enabled)
    with open(console.dolphin_home_path + 'GameSettings/GALE01.ini', 'w') as f:
        f.write("\n".join(new))

    console.run(iso_path='../../isos/game.iso')
    console.connect()
    while True:
        _ = console.step()