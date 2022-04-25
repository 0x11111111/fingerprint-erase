import os
import ffmpeg
import multiprocessing as mp
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


# def pic2video_clip(group_number):
#     with open('info.json', 'r') as f:
#         info = json.load(f, object_hook=lambda x: SimpleNamespace(**x))
#         f.close()
#
#     frame_path = os.path.join(info.folder, str(group_number))
#     frame_path_list = sorted(os.listdir(frame_path))
#     frame_full_path_list = [frame_path + '/' + frame_name for frame_name in frame_path_list]
#
#     # print(frame_full_path_list)
#     clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(frame_full_path_list, fps=info.fps)
#     return group_number, clip

def pic2video_clip(group_number):
    with open('info.json', 'r') as f:
        info = json.load(f, object_hook=lambda x: SimpleNamespace(**x))
        f.close()

    option = info.option
    codec = option.codec
    threads = mp.cpu_count()
    preset = option.preset
    frame_path = os.path.abspath(os.path.join(info.folder, str(group_number), '*.jpeg'))
    tmp_output_path = os.path.abspath(os.path.join(info.folder, '{}.mp4'.format(group_number)))
    print(tmp_output_path)
    test = (
        ffmpeg
        .input(frame_path, pattern_type='glob', framerate=info.fps)
        .output(tmp_output_path, vcodec=codec, threads=threads, preset=preset)
        .global_args('-loglevel', 'quiet')

    )
    print(test.get_args())
    test.run(overwrite_output=True)
    # print(frame_full_path_list)
    # clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(frame_full_path_list, fps=info.fps)
    return group_number, tmp_output_path
