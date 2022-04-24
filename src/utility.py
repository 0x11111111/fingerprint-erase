import os
import moviepy
import moviepy.video.io.ImageSequenceClip
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


def pic2video(frame_path, video_dst, codec, fps):
    frame_name = sorted(os.listdir(frame_path))
    frame_path_list = [frame_path + frame_name for frame_name in frame_name]
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(frame_path_list, fps=fps)
    clip.write_videofile(video_dst, codec='libx264')
