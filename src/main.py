import os
import time
import cv2
import threadpool
import json
import ffmpeg
import platform
import sys
import multiprocessing

from types import SimpleNamespace
from gui_get_option import get_option
from delegation_multi_process import multi_process_fingerprint_erase
from delegation_realtime_process import realtime_process
from utility import sn2dict


if __name__ == '__main__':
    info = SimpleNamespace()
    option = None
    batch_run = len(sys.argv) > 1

    if batch_run:
        # Options fetched from file in argv[1]
        with open(sys.argv[1], 'r') as f:
            info = json.load(f, object_hook=lambda x: SimpleNamespace(**x))
            option = info.option
            f.close()

    else:
        # No command line arguments
        option = get_option()
        option = SimpleNamespace(**option)
        info.option = option

    # Spawn a new temporary file folder
    info.folder = os.path.join('../.tmp', '{}'.format(int(time.time() * 1000)))
    if not os.path.exists("../.tmp"):
        os.mkdir("../.tmp")
    os.mkdir(info.folder)

    info.flags = SimpleNamespace(
        # 核心指纹圈
        circle_on=False,
        # 关键点连线
        connection_on=False,
        # 坐标显示
        coordination_on=False,
        # 指纹核心参数
        output_on=False,
        # 手掌朝向
        orientation_on=False,
        # 帧率
        frame_rate_on=False,
        # 手掌包络框
        box_on=False,
    )

    info.EPS = 0.0001
    info.tip_dip_length_ratio = 0.7
    info.pinky_ring_width_ratio = 0.89
    info.thumb_width_length_ratio = 0.62
    info.finger_mcp_width_ratio = 0.65
    info.landmark_order = 'abcdefghijklmnopqrstu'

    performance_attributes = SimpleNamespace(
        time_initial_start=int(time.time())
    )
    if batch_run:
        print('Initial time: {}'.format(
            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(performance_attributes.time_initial_start))))

    info.file_path = option.file_path
    # Kernel size of Gaussian should be odd only
    info.kernel_size = int(option.blur_value // 2 * 2 + 1)
    info.camera_input_no = int(option.camera_input_no)

    info.blur_mode = 'nope'
    if option.averaging:
        info.blur_mode = 'averaging'
    elif option.gaussian:
        info.blur_mode = 'gaussian'
    elif option.median:
        info.blur_mode = 'median'
    elif option.bilateral:
        info.blur_mode = 'bilateral'

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

    performance_attributes.time_erase_start = int(time.time())

    if info.camera_input_no < 0:
        # Input from video file
        cap = cv2.VideoCapture(info.file_path)
        info.frame_count = performance_attributes.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        performance_attributes.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        performance_attributes.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        performance_attributes.frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
        cap.release()

        performance_attributes.threads = info.num_processes = os.cpu_count()
        if option.single_process or option.single_thread:
            performance_attributes.threads = info.num_processes = 1
        info.frame_jump_unit = info.frame_count // info.num_processes

        info_dict = sn2dict(info)
        with open('./info.json', 'w') as f:
            json.dump(info_dict, f)
            f.close()

        if batch_run:
            print('Erase start time: {}'.format(
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(performance_attributes.time_erase_start))))
            print('Time elapsed: {}s'.format(
                performance_attributes.time_erase_start - performance_attributes.time_initial_start))

        if option.multi_process or option.single_process:
            pool = multiprocessing.Pool(info.num_processes)
            pool.map_async(multi_process_fingerprint_erase, range(info.num_processes))
            pool.close()
            pool.join()

        elif option.multi_thread or option.single_thread:
            pool = threadpool.ThreadPool(info.num_processes)
            requests = threadpool.makeRequests(multi_process_fingerprint_erase, range(info.num_processes))
            [pool.putRequest(req) for req in requests]
            pool.wait()

        source_video = ffmpeg.input(option.file_path)
        audio = source_video.audio

    else:
        # Input from camera stream.
        cap = cv2.VideoCapture(info.camera_input_no)
        performance_attributes.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        performance_attributes.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        info_dict = sn2dict(info)
        with open('./info.json', 'w') as f:
            json.dump(info_dict, f)
            f.close()

        temp_path, audio_path, record_time, processed_frame_count = realtime_process()
        performance_attributes.frame_count = processed_frame_count
        performance_attributes.frame_rate = int((processed_frame_count / record_time) * 100) / 100
        audio = ffmpeg.input(audio_path).audio

    performance_attributes.time_concat_start = int(time.time())
    if batch_run:
        print('Concatenate start time: {}'.format(
            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(performance_attributes.time_concat_start))))
        print('Time elapsed: {}s'.format(
            performance_attributes.time_concat_start - performance_attributes.time_initial_start))

    frame_path = os.path.abspath(info.folder)
    dst_path = './output'
    if not os.path.exists(dst_path):
        os.mkdir(dst_path)
    output_file_path = os.path.abspath(os.path.join(dst_path, 'erased' + video_extension))
    output_video = None

    if platform.system() == 'Windows':
        # Windows doesn't support POSIX like wildcard.
        file_list = os.listdir(frame_path)
        i = 0
        for name in file_list:
            # Skip WAV files.
            if not name[-5:] == '.jpeg':
                continue

            src = os.path.join(frame_path, name)
            dst = os.path.join(frame_path, 'image{0:08d}.jpeg'.format(i))
            os.rename(src, dst)
            i += 1

        output_video = (
            ffmpeg
            .input(os.path.join(frame_path, 'image%08d.jpeg'), framerate=performance_attributes.frame_rate)
        )

    else:
        # Unix-like is GREAT.
        output_video = (
            ffmpeg
            .input(os.path.join(frame_path, '*.jpeg'), pattern_type='glob', framerate=performance_attributes.frame_rate)
        )

    output_video = (
        ffmpeg
        .concat(output_video, audio, v=1, a=1)
        .output(output_file_path, vcodec=codec, threads=os.cpu_count(), preset=preset)
        # .global_args('-loglevel', 'quiet')
    )

    performance_attributes.time_output_start = int(time.time())
    if batch_run:
        print('Output start time: {}'.format(
            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(performance_attributes.time_output_start))))
        print('Time elapsed: {}s'.format(
            performance_attributes.time_output_start - performance_attributes.time_initial_start))

    output_video.run(overwrite_output=True)

    performance_attributes.time_finish = int(time.time())

    if info.camera_input_no < 0:
        print('ALL ACCOMPLISHED')
        print('Output file: {}'.format(output_file_path))
        print('Stage time consumption:')
        print('- Erasing fingertips: {}s'.format(
            performance_attributes.time_concat_start - performance_attributes.time_erase_start))
        print('- Concatenating each images: {}s'.format(
            performance_attributes.time_output_start - performance_attributes.time_concat_start))
        print('- Output final video: {}s'.format(
            performance_attributes.time_finish - performance_attributes.time_output_start))
        print('Total: {}s'.format(performance_attributes.time_finish - performance_attributes.time_initial_start))
        print('Process information:')
        print('- Threads/Cores utilized: {}'.format(performance_attributes.threads))
        print('- Frames processed: {}'.format(performance_attributes.frame_count))
        print('- Frames size: {} x {}'.format(performance_attributes.frame_width, performance_attributes.frame_height))
        print('- Video frame rate: {:.3f}FPS'.format(performance_attributes.frame_rate))
        print('- Average frame processing rate: {:.3f}FPS'.format(performance_attributes.frame_count / (
            performance_attributes.time_concat_start - performance_attributes.time_erase_start)))

    else:
        print('ALL ACCOMPLISHED')
        print('Output file: {}'.format(output_file_path))
        print('Stage time consumption:')
        print('- Recording fingertips: {}s'.format(
            performance_attributes.time_concat_start - performance_attributes.time_erase_start))
        print('- Concatenating each images: {}s'.format(
            performance_attributes.time_output_start - performance_attributes.time_concat_start))
        print('- Output final video: {}s'.format(
            performance_attributes.time_finish - performance_attributes.time_output_start))
        print('Total: {}s'.format(performance_attributes.time_finish - performance_attributes.time_initial_start))
        print('Process information:')
        print('- Frames processed: {}'.format(performance_attributes.frame_count))
        print('- Frames size: {} x {}'.format(performance_attributes.frame_width, performance_attributes.frame_height))
        print('- Average frame processing rate: {:.3f}FPS'.format(performance_attributes.frame_count / (
            performance_attributes.time_concat_start - performance_attributes.time_erase_start)))