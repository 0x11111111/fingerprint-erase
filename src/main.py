import mediapipe
import os
import cv2
import time
import multiprocessing as mp
import threadpool
import json
from functools import partial
from types import SimpleNamespace
from moviepy.editor import VideoFileClip

from gui_get_option import get_option
from core_neomask_finger_tracking import fingerprint_erase
from utility import sn2dict, pic2video


if __name__ == '__main__':
    args = SimpleNamespace(
        folder=os.path.join('../.tmp', '{}'.format(int(round(time.time() * 1000))))
    )

    args.debug_mode = SimpleNamespace(
        circle_on=False,
        landmark_on=False,
        coordination_on=False,
        output_on=False,
        orientation_on=False,
        frame_rate_on=True,
        scoop_on=False,
    )
    if not os.path.exists("../.tmp"):
        os.mkdir("../.tmp")

    os.mkdir(args.folder)

    args.EPS = 0.0001
    args.fingertip_radius_sn = SimpleNamespace(
        thumb=20.0,
        index=18.4,
        middle=18.3,
        ring=17.3,
        pinky=15.3
    )
    args.finger_length_sn = SimpleNamespace(
        thumb=32.1,
        index=24.7,
        middle=26.4,
        ring=26.3,
        pinky=23.7
    )
    args.tip_dip_length_ratio = 0.8
    args.pinky_ring_width_ratio = 0.89
    args.thumb_width_length_ratio = 0.60
    args.finger_mcp_width_ratio = 0.65

    args.landmark_order = 'abcdefghijklmnopqrstu'

    selection = get_option()
    args.file_path = selection['file_path']
    # Kernel size of Gaussian should be odd only
    args.kernel_size = int(selection['blur_value'] // 2 * 2 + 1)

    args.blur_mode = 0
    if selection['normalization']:
        args.blur_mode = 1
    elif selection['gaussian']:
        args.blur_mode = 2

    # cap = cv2.VideoCapture(args.file_path)
    # frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # cap.release()

    video = VideoFileClip(args.file_path)
    frame_count = video.reader.nframes
    frame_rate = video.fps
    audio = video.audio


    video.close()

    args.num_processes = mp.cpu_count()
    args.frame_jump_unit = frame_count // args.num_processes

    args_dict = sn2dict(args)

    with open('./args.json', 'w') as f:
        json.dump(args_dict, f)

    # p = mp.Pool(args.num_processes)
    # p.map(fingerprint_erase, range(args.num_processes))

    pool = threadpool.ThreadPool(args.num_processes)
    requests = threadpool.makeRequests(fingerprint_erase, range(args.num_processes))
    [pool.putRequest(req) for req in requests]
    pool.wait()
