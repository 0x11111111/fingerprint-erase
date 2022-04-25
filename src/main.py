import os
import time
import cv2
import threadpool
import json
import ffmpeg
import subprocess
import platform
import math
import sys
import multiprocessing as mp

from types import SimpleNamespace
from moviepy.editor import VideoFileClip, concatenate_videoclips, CompositeVideoClip

from gui_get_option import get_option
from core_neomask_finger_tracking import fingerprint_erase
from utility import sn2dict, pic2video_clip

clip_list = []


# Callback function to retrieve results from threadpool results.
def get_clip_result(requests, res):
    global clip_list
    clip_list.append(res)


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
    info.fingertip_radius_sn = SimpleNamespace(
        thumb=20.0,
        index=18.4,
        middle=18.3,
        ring=17.3,
        pinky=15.3
    )
    info.finger_length_sn = SimpleNamespace(
        thumb=32.1,
        index=24.7,
        middle=26.4,
        ring=26.3,
        pinky=23.7
    )
    info.tip_dip_length_ratio = 0.8
    info.pinky_ring_width_ratio = 0.89
    info.thumb_width_length_ratio = 0.60
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

    codec = None
    video_extension = None
    # 'mpeg4': True, 'libx264': False, 'rawvideo': False, 'png': False, 'libvpx': False, 'libvorbis': False
    if option.mpeg4:
        codec = 'mpeg4'
        video_extension = '.mp4'
    elif option.libx264:
        codec = 'libx264'
        video_extension = '.mp4'
    elif option.rawvideo:
        codec = 'rawvideo'
        video_extension = '.avi'
    elif option.png:
        codec = 'png'
        video_extension = '.avi'
    elif option.libvpx:
        codec = 'libvpx'
        video_extension = '.webm'
    elif option.libvorbis:
        codec = 'libvorbis'
        video_extension = '.ogv'

    cap = cv2.VideoCapture(info.file_path)
    frame_count = performance_attributes.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    performance_attributes.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    performance_attributes.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    info.fps = frame_rate = performance_attributes.frame_rate = int(cap.get(cv2.CAP_PROP_FPS))

    video = VideoFileClip(info.file_path)
    # frame_count = video.reader.nframes
    # frame_rate = video.fps
    # info.fps = frame_rate
    audio = video.audio

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
    pool = threadpool.ThreadPool(info.num_processes)
    requests = threadpool.makeRequests(pic2video_clip, range(info.num_processes), get_clip_result)
    [pool.putRequest(req) for req in requests]
    pool.wait()

    clip_list.sort(key=lambda x: x[0])
    # print(clip_list)
    clip_to_concat = [tup[1] for tup in clip_list]
    # print(clip_to_concat)

    performance_attributes.time_concat_start = int(time.time())
    print('Concatenate start time: {}'.format(
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(performance_attributes.time_concat_start))))
    print('Time elapsed: {}s'.format(
        performance_attributes.time_concat_start - performance_attributes.time_initial_start))
    concat = concatenate_videoclips(clip_to_concat)
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = frame_rate = cap.get(cv2.CAP_PROP_FPS)
    duration = math.floor(frames / fps)
    concat = concat.subclip(0, duration)
    # Get rid of the last frame which may cause IndexError Exception. Bug caused by moviepy
    concat = concat.subclip(t_end=(concat.duration - 10.0 / concat.fps))

    performance_attributes.time_output_start = int(time.time())
    print('Output start time: {}'.format(
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(performance_attributes.time_output_start))))
    print('Time elapsed: {}s'.format(
        performance_attributes.time_output_start - performance_attributes.time_initial_start))
    dst_path = '.'
    audio = audio.subclip(0, concat.duration)
    audio.write_audiofile(
        filename=dst_path + '/output' + '.wav',
    )

    concat.set_audio(audio)
    if not os.path.exists(dst_path):
        os.mkdir(dst_path)
    concat.write_videofile(
        filename=dst_path + '/output' + video_extension,
        codec=codec,
        threads=info.num_processes
    )
    cap.release()
    video.close()

    video_file_path = os.path.abspath(dst_path + '/output' + video_extension)
    audio_file_path = os.path.abspath(dst_path + '/output' + '.wav')
    output_file_path = os.path.abspath(dst_path + '/blurred' + video_extension)
    if platform.system() == 'Linux':
        input_video = ffmpeg.input(video_file_path)
        input_audio = ffmpeg.input(audio_file_path)
        (
            ffmpeg
            .concat(input_video, input_audio, v=1, a=1)
            .output(output_file_path)
            .run(overwrite_output=True)
        )
    else:
        # Windows encounters file path errors.
        subprocess.run(f"ffmpeg -i {video_file_path} -i {audio_file_path} -y {output_file_path}")

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
