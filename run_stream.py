from training.arena import Arena

from functools import partial
import fire


if __name__ == '__main__':
    fire.Fire(partial(Arena, obs_streaming=True, test=True))