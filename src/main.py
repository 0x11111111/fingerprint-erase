import os
import time
import cv2
import threadpool
import json
import platform
import ffmpeg
import subprocess
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
    args = SimpleNamespace(
        folder=os.path.join('../.tmp', '{}'.format(int(round(time.time()))))
    )

    args.debug_mode = SimpleNamespace(
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
    performance_attributes = SimpleNamespace(
        time_initial_start=int(time.time())
    )
    print('Initial time: {}'.format(
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(performance_attributes.time_initial_start))))

    # print(selection)
    selection = SimpleNamespace(**selection)
    args.file_path = selection.file_path
    # Kernel size of Gaussian should be odd only
    args.kernel_size = int(selection.blur_value // 2 * 2 + 1)

    args.blur_mode = 0
    if selection.normalization:
        args.blur_mode = 1
    elif selection.gaussian:
        args.blur_mode = 2

    codec = None
    video_extension = None
    # 'mpeg4': True, 'libx264': False, 'rawvideo': False, 'png': False, 'libvpx': False, 'libvorbis': False
    if selection.mpeg4:
        codec = 'mpeg4'
        video_extension = '.mp4'
    elif selection.libx264:
        codec = 'libx264'
        video_extension = '.mp4'
    elif selection.rawvideo:
        codec = 'rawvideo'
        video_extension = '.avi'
    elif selection.png:
        codec = 'png'
        video_extension = '.avi'
    elif selection.libvpx:
        codec = 'libvpx'
        video_extension = '.webm'
    elif selection.libvorbis:
        codec = 'libvorbis'
        video_extension = '.ogv'

    cap = cv2.VideoCapture(args.file_path)
    performance_attributes.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    performance_attributes.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    performance_attributes.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    performance_attributes.frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    cap.release()

    video = VideoFileClip(args.file_path)
    frame_count = video.reader.nframes
    frame_rate = video.fps
    args.fps = frame_rate
    audio = video.audio

    performance_attributes.threads = args.num_processes = mp.cpu_count()
    if selection.single_thread:
        performance_attributes.threads = args.num_processes = 1
    args.frame_jump_unit = frame_count // args.num_processes

    args_dict = sn2dict(args)

    with open('./args.json', 'w') as f:
        json.dump(args_dict, f)

    performance_attributes.time_erase_start = int(time.time())
    print('Erase start time: {}'.format(
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(performance_attributes.time_erase_start))))
    print('Time elapsed: {}s'.format(
        performance_attributes.time_erase_start - performance_attributes.time_initial_start))
    operation_system = platform.system()
    if operation_system == 'Linux':
        pool = mp.Pool(args.num_processes)
        pool.map(fingerprint_erase, range(args.num_processes))
        pool.close()
        pool.join()

    else:
        # Windows doesn't support fork()
        pool = threadpool.ThreadPool(args.num_processes)
        requests = threadpool.makeRequests(fingerprint_erase, range(args.num_processes))
        [pool.putRequest(req) for req in requests]
        pool.wait()

    performance_attributes.time_compile_start = int(time.time())
    print('Compile start time: {}'.format(
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(performance_attributes.time_compile_start))))
    print('Time elapsed: {}s'.format(
        performance_attributes.time_compile_start - performance_attributes.time_initial_start))
    pool = threadpool.ThreadPool(args.num_processes)
    requests = threadpool.makeRequests(pic2video_clip, range(args.num_processes), get_clip_result)
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

    performance_attributes.time_output_start = int(time.time())
    print('Output start time: {}'.format(
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(performance_attributes.time_output_start))))
    print('Time elapsed: {}s'.format(
        performance_attributes.time_output_start - performance_attributes.time_initial_start))
    dst_path = '.'
    audio = audio.subclip(0, concat.duration)
    audio.write_audiofile(
        filename=dst_path + '/output' + '.wav'
    )

    concat.set_audio(audio)
    if not os.path.exists(dst_path):
        os.mkdir(dst_path)
    concat.write_videofile(
        filename=dst_path + '/output' + video_extension,
        codec=codec,
        threads=args.num_processes
    )
    video.close()

    video_file_path = os.path.abspath(dst_path + '/output' + video_extension)
    audio_file_path = os.path.abspath(dst_path + '/output' + '.wav')
    output_file_path = os.path.abspath(dst_path + '/blurred' + video_extension)
    if operation_system == 'Linux':
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
