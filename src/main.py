import os
import time
import cv2
import threadpool
import json
import ffmpeg
import platform
import sys
import multiprocessing as mp

from types import SimpleNamespace
from gui_get_option import get_option
from core_neomask_finger_tracking import fingerprint_erase
from utility import sn2dict


if __name__ == '__main__':
    info = SimpleNamespace()
    option = None
    if len(sys.argv) <= 1:
        # No command line arguments
        option = get_option()
        option = SimpleNamespace(**option)
        info.option = option
    else:
        # Options fetched from file in argv[1]
        with open(sys.argv[1], 'r') as f:
            info = json.load(f, object_hook=lambda x: SimpleNamespace(**x))
            option = info.option
            f.close()
    # Spawn a new temporary file folder
    info.folder = os.path.join('../.tmp', '{}'.format(int(round(time.time() * 1000))))

    info.debug_mode = SimpleNamespace(
        circle_on=False,
        landmark_on=False,
        coordination_on=False,
        output_on=False,
        orientation_on=False,
        frame_rate_on=False,
        scoop_on=False,
    )

    if not os.path.exists("../.tmp"):
        os.mkdir("../.tmp")
    os.mkdir(info.folder)

    info.EPS = 0.0001
    info.tip_dip_length_ratio = 0.7
    info.pinky_ring_width_ratio = 0.89
    info.thumb_width_length_ratio = 0.62
    info.finger_mcp_width_ratio = 0.65

    info.landmark_order = 'abcdefghijklmnopqrstu'

    performance_attributes = SimpleNamespace(
        time_initial_start=int(time.time())
    )
    print('Initial time: {}'.format(
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(performance_attributes.time_initial_start))))

    info.file_path = option.file_path
    # Kernel size of Gaussian should be odd only
    info.kernel_size = int(option.blur_value // 2 * 2 + 1)

    info.blur_mode = 0
    if option.normalization:
        info.blur_mode = 1
    elif option.gaussian:
        info.blur_mode = 2

    video_extension = '.mp4'

    preset = None
    if option.ultrafast:
        preset = 'ultrafast'
    elif option.superfast:
        preset = 'superfast'
    elif option.faster:
        preset = 'faster'
    elif option.fast:
        preset = 'fast'
    elif option.medium:
        preset = 'medium'
    elif option.slow:
        preset = 'slow'
    elif option.slower:
        preset = 'slower'
    elif option.veryslow:
        preset = 'veryslow'
    option.preset = preset

    codec = None
    if option.h265:
        codec = 'libx265'
    elif option.h264:
        codec = 'h264'
    option.codec = codec

    cap = cv2.VideoCapture(info.file_path)
    frame_count = performance_attributes.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    performance_attributes.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    performance_attributes.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    info.fps = frame_rate = performance_attributes.frame_rate = int(cap.get(cv2.CAP_PROP_FPS))

    performance_attributes.threads = info.num_processes = mp.cpu_count()
    if option.single_process or option.single_thread:
        performance_attributes.threads = info.num_processes = 1
    info.frame_jump_unit = frame_count // info.num_processes

    info_dict = sn2dict(info)
    with open('./info.json', 'w') as f:
        json.dump(info_dict, f)
        f.close()

    performance_attributes.time_erase_start = int(time.time())
    print('Erase start time: {}'.format(
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(performance_attributes.time_erase_start))))
    print('Time elapsed: {}s'.format(
        performance_attributes.time_erase_start - performance_attributes.time_initial_start))

    if option.multi_process or option.single_process:
        pool = mp.Pool(info.num_processes)
        pool.map_async(fingerprint_erase, range(info.num_processes))
        pool.close()
        pool.join()

    else:
        pool = threadpool.ThreadPool(info.num_processes)
        requests = threadpool.makeRequests(fingerprint_erase, range(info.num_processes))
        [pool.putRequest(req) for req in requests]
        pool.wait()

    performance_attributes.time_compile_start = int(time.time())
    print('Compile start time: {}'.format(
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(performance_attributes.time_compile_start))))
    print('Time elapsed: {}s'.format(
        performance_attributes.time_compile_start - performance_attributes.time_initial_start))

    source_video = ffmpeg.input(option.file_path)
    audio = source_video.audio

    performance_attributes.time_concat_start = int(time.time())
    print('Concatenate start time: {}'.format(
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(performance_attributes.time_concat_start))))
    print('Time elapsed: {}s'.format(
        performance_attributes.time_concat_start - performance_attributes.time_initial_start))

    frame_path = os.path.abspath(info.folder)
    dst_path = '.'
    if not os.path.exists(dst_path):
        os.mkdir(dst_path)
    output_file_path = os.path.abspath(os.path.join(dst_path, 'erased' + video_extension))
    output_video = None
    if platform.system() == 'Windows':
        file_list = os.listdir(frame_path)
        for i, name in enumerate(file_list):
            src = os.path.join(frame_path, name)
            dst = os.path.join(frame_path, 'image{0:08d}.jpeg'.format(i))
            os.rename(src, dst)

        output_video = (
            ffmpeg
            .input(os.path.join(frame_path, 'image%08d.jpeg'), framerate=info.fps)

        )
    else:
        output_video = (
            ffmpeg
            .input(os.path.join(frame_path, '*.jpeg'), pattern_type='glob', framerate=info.fps)
        )

    output_video = (
        ffmpeg
        .concat(output_video, audio, v=1, a=1)
        .output(output_file_path, vcodec=codec, threads=mp.cpu_count(), preset=preset)
        .global_args('-loglevel', 'quiet')
    )

    performance_attributes.time_output_start = int(time.time())
    print('Output start time: {}'.format(
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(performance_attributes.time_output_start))))
    print('Time elapsed: {}s'.format(
        performance_attributes.time_output_start - performance_attributes.time_initial_start))

    output_video.run(overwrite_output=True)
    cap.release()

    performance_attributes.time_finish = int(time.time())
    print('ALL ACCOMPLISHED')
    print('Output file: {}'.format(output_file_path))
    print('Stage time consumption:')
    print('- Erasing fingertips: {}s'.format(
        performance_attributes.time_compile_start - performance_attributes.time_erase_start))
    print('- Compiling frames into video clips: {}s'.format(
        performance_attributes.time_concat_start - performance_attributes.time_compile_start))
    print('- Concatenating each video clips: {}s'.format(
        performance_attributes.time_output_start - performance_attributes.time_concat_start))
    print('- Output final video: {}s'.format(
        performance_attributes.time_finish - performance_attributes.time_output_start))
    print('Total: {}s'.format(performance_attributes.time_finish - performance_attributes.time_initial_start))
    print('Process information:')
    print('- Threads/Cores utilized: {}'.format(performance_attributes.threads))
    print('- Frames processed: {}'.format(performance_attributes.frame_count))
    print('- Frames size: {} x {}'.format(performance_attributes.frame_width, performance_attributes.frame_height))
    print('- Video frame rate: {}FPS'.format(performance_attributes.frame_rate))
    print('- Average frame processing rate: {:.3f}FPS'.format(performance_attributes.frame_count / (
            performance_attributes.time_compile_start - performance_attributes.time_erase_start)))