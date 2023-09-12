import time
import numpy as np
from openal import * 
from itertools import cycle

if __name__ == "__main__":
    x_pos = 5
    sleep_time = 5
    source = oalOpen("CantinaBand60.wav")
    source.set_position([0, 0, 0])
    source.set_looping(True)
    source.play()
    listener = Listener()
    listener.set_position([0, 0, 0])

    pos = cycle(np.linspace(0, 360, sleep_time))
    print(np.linspace(0, 360, sleep_time))
    i=0
    while source.get_state() == AL_PLAYING:
        n = np.radians(next(pos))
        source.set_position((0, 0, i))
        source.set_direction()
        # source.set_position([np.cos(n) * 5, np.sin(n) * 2, np.sin(n) * 2])
        print("Playing at: {0}".format(source.position))
        time.sleep(sleep_time)
        x_pos *= -1
        i+=1
    
    oalQuit()


# import time
# import math
# from openal.audio import SoundSink, SoundSource
# from openal.loaders import load_wav_file

# if __name__ == "__main__":
#     sink = SoundSink()
#     sink.activate()
#     source = SoundSource(position=[0, 0, 0])
#     source.looping = True
#     data = load_wav_file("tone5.wav")
#     source.queue(data)
#     sink.play(source)
#     t = 0
#     while True:
#         y_pos = 5*math.sin(math.radians(t))
#         x_pos = 5*math.cos(math.radians(t))
#         source.position = [x_pos, y_pos, 0]
#         sink.update()
#         print("playing at %r" % source.position)
#         time.sleep(0.1)
#         t += 5
