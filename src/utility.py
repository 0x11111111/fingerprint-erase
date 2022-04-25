import os
import moviepy
import moviepy.video.io.ImageSequenceClip
import json
from types import SimpleNamespace


def sn2dict(sn):
    d = dict()
    if isinstance(sn, SimpleNamespace):
        for k, v in sn.__dict__.items():
            if isinstance(v, SimpleNamespace):
                d[k] = sn2dict(v)
            else:
                d[k] = v

    return d


def pic2video_clip(group_number):
    with open('info.json', 'r') as f:
        info = json.load(f, object_hook=lambda x: SimpleNamespace(**x))
        f.close()

    frame_path = os.path.join(info.folder, str(group_number))
    frame_path_list = sorted(os.listdir(frame_path))
    frame_full_path_list = [frame_path + '/' + frame_name for frame_name in frame_path_list]

    # print(frame_full_path_list)
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(frame_full_path_list, fps=info.fps)
    return group_number, clip


