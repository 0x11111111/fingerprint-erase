import json
import os
import time
import types
import wave
from multiprocessing import Process, Event, Manager
from types import SimpleNamespace

import PySimpleGUI as sg
import ffmpeg
import mediapipe
import numpy as np
import pyaudio
import sounddevice as sd
from cv2 import cv2

from core_finger_processor import preprocess, detect_orientation, detect_finger_self_occlusion, detect_palm_occlusion, \
    process_fingertip


def realtime_process() -> ():
    """Delegation for realtime camera captured video stream process.

    Executes two subprocess handling video and audio recording respectively. Video is captured via camera indicated
    by info.json.

    Returns:
        tuple: tuple of temporary folder path, output audio path, record time and frame count
    """
    finished = Event()
    return_dict = Manager().dict()

    info = None
    if not __name__ == '__main__' and os.path.exists('./info.json'):
        with open('./info.json', 'r') as f:
            info = json.load(f, object_hook=lambda x: SimpleNamespace(**x))
            f.close()
        info.normalized_kernel = np.array(info.normalized_kernel)

    else:
        info = SimpleNamespace(
            folder=os.path.join('../.tmp', '{}'.format(int(time.time() * 1000))),
            flags=SimpleNamespace(
                circle_on=False,
                connection_on=False,
                coordination_on=False,
                output_on=False,
                orientation_on=False,
                frame_rate_on=False,
                box_on=False,
            ),
            kernel_size=11,
            option=SimpleNamespace(
                random=True,
                averaging=False,
                gaussian=False,
                median=False,
                bilateral=False,
                nope=False
            ),
            blur_mode='averaging',
            camera_input_no=0
        )

        info.EPS = 0.0001
        info.tip_dip_length_ratio = 0.7
        info.pinky_ring_width_ratio = 0.89
        info.thumb_width_length_ratio = 0.62
        info.finger_mcp_width_ratio = 0.65
        info.landmark_order = 'abcdefghijklmnopqrstu'

    erase = Process(target=fingerprint_erase, args=(int(info.camera_input_no), finished, return_dict, info))
    recorder = Process(target=sound_recorder, args=(finished, return_dict, info))
    recorder.start()
    erase.start()

    erase.join()
    recorder.join()

    time_list = []
    erase_start = return_dict['fingerprint_erase_start_timestamp']
    offset_time = erase_start - return_dict['sound_recorder_start_timestamp']
    source_audio_path = return_dict['record_path']
    output_audio_path = os.path.abspath(os.path.join(info.folder, 'concatenated.wav'))
    record_time = 0.0

    for timestamp_tuple in return_dict['fingerprint_timestamp_list']:
        time_list.append(
            (
                (timestamp_tuple[0] - erase_start + offset_time) / 1000.0,
                (timestamp_tuple[1] - erase_start + offset_time) / 1000.0
            )
        )
        record_time += time_list[-1][1] - time_list[-1][0]

    if time_list:
        audio = ffmpeg.input(source_audio_path, ss=time_list[0][0], to=time_list[0][1])
        for i in range(1, len(time_list)):
            to_concat = ffmpeg.input(source_audio_path, ss=time_list[i][0], to=time_list[i][1])
            audio = ffmpeg.concat(audio, to_concat, v=0, a=1)

        audio.output(output_audio_path).run(overwrite_output=True)

        return info.folder, output_audio_path, record_time, return_dict['frame_count']

    else:
        return tuple()


def fingerprint_erase(video_source: int, finished_event: Event, ret_dict: dict, info: types.SimpleNamespace):
    """Handling camera input stream and fingerprint erasure.

    This function is called by realtime_process. Video input stream is indicated by video_source. Interprocess
    communication is indicated by finished_event.

    Args:
        video_source (int): indicates which camera is activated
        finished_event (Event): sets the event if the stop is called in video recording and stops sound recording in
            sound_recorder()
        ret_dict (dict): video records timestamp and frame count
        info (types.SimpleNamespace): consts and attributes from main

    Returns:

    """
    temp_path = info.folder
    if not os.path.exists(temp_path):
        os.mkdir(temp_path)

    cap = cv2.VideoCapture(video_source)

    mp_drawing = mediapipe.solutions.drawing_utils
    mp_hands = mediapipe.solutions.hands

    layout = [
        [sg.Image(filename='', key='image')],
        [
            sg.Button(key='Erase', button_text='抹除', size=(10, 1), font='Any 15'),
            sg.Button(key='Exit', button_text='退出', size=(10, 1), font='Any 15')
        ],
        [
            sg.Checkbox(key='flip', text='水平翻转', font='Any 15'),
            sg.Checkbox(key='circle_on', text='核心指纹圈', font='Any 15'),
            sg.Checkbox(key='connection_on', text='关键点连线', font='Any 15'),
            sg.Checkbox(key='output_on', text='指纹核心参数', font='Any 15'),
            sg.Checkbox(key='orientation_on', text='手掌朝向', font='Any 15'),
            sg.Checkbox(key='frame_rate_on', text='帧率', font='Any 15'),
            sg.Checkbox(key='box_on', text='手掌包络框', font='Any 15'),
        ],
        [
            sg.Text('模糊程度', font='Any 15'),
            sg.Slider(key='blur_value', range=(1, 51), default_value=info.kernel_size, size=(90, 15),
                      orientation='horizontal')
        ],
        [
            sg.Text('模糊处理', font='Any 15'),
            sg.Radio(key='random', text='随机模糊', group_id='blur_option', default=info.option.random, font='Any 15'),
            sg.Radio(key='averaging', text='平均平滑', group_id='blur_option', default=info.option.averaging,
                     font='Any 15'),
            sg.Radio(key='gaussian', text='高斯模糊', group_id='blur_option', default=info.option.gaussian, font='Any 15'),
            sg.Radio(key='median', text='中值滤波', group_id='blur_option', default=info.option.median, font='Any 15'),
            sg.Radio(key='bilateral', text='双边滤波', group_id='blur_option', default=info.option.bilateral,
                     font='Any 15'),
            sg.Radio(key='nope', text='不处理', group_id='blur_option', default=info.option.nope, font='Any 15')
        ]
    ]

    window = sg.Window('实时抹除', layout, location=(800, 400))
    erase_flag = False
    start_timestamp = 0
    end_timestamp = 0
    timestamp_list = []
    processed_frame_count = 0
    prev_frame_time = 0
    curr_frame_time = 0
    last_kernel_size = 0
    normalized_kernel = None
    ret_dict['fingerprint_erase_start_timestamp'] = int(time.time() * 1000)

    try:
        with mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.6) as hands:
            while True:
                event, values = window.read(timeout=100)
                option = SimpleNamespace(**values)

                if event == 'Exit' or event == sg.WIN_CLOSED:
                    if erase_flag:
                        end_timestamp = int(time.time() * 1000)
                        timestamp_list.append((start_timestamp, end_timestamp))

                    break

                elif event == 'Erase':
                    if not erase_flag:
                        start_timestamp = int(time.time() * 1000)
                        erase_flag = True
                        window['Erase'].update(text='停止')

                    else:
                        end_timestamp = int(time.time() * 1000)
                        timestamp_list.append((start_timestamp, end_timestamp))
                        erase_flag = False
                        window['Erase'].update(text='抹除')

                if erase_flag:
                    _, frame = cap.read()
                    if option.flip:
                        frame = cv2.flip(frame, 1)

                    kernel_size = int(option.blur_value)

                    info.flags.circle_on = option.circle_on
                    info.flags.connection_on = option.connection_on
                    info.flags.output_on = option.output_on
                    info.flags.orientation_on = option.orientation_on
                    info.flags.frame_rate_on = option.frame_rate_on
                    info.flags.box_on = option.box_on
                    info.kernel_size = int(option.blur_value // 2 * 2 + 1)
                    info.blur_mode = 'nope'
                    if option.random:
                        info.blur_mode = 'random'
                        if not kernel_size == last_kernel_size:
                            last_kernel_size = kernel_size
                            # Generate new filter kernel since kernel size has been changed.
                            kernel = np.random.randint(1, kernel_size ** 4 + 1, size=[kernel_size] * 2)
                            sum_kernel = np.sum(kernel)
                            normalized_kernel = kernel / sum_kernel

                    elif option.averaging:
                        info.blur_mode = 'averaging'
                    elif option.gaussian:
                        info.blur_mode = 'gaussian'
                        kernel_size = int(kernel_size // 2 * 2 + 1)
                    elif option.median:
                        info.blur_mode = 'median'
                        kernel_size = int(kernel_size // 2 * 2 + 1)
                    elif option.bilateral:
                        info.blur_mode = 'bilateral'

                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image.flags.writeable = False
                    results = hands.process(image)
                    image.flags.writeable = True
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                    # Rendering results
                    if results.multi_hand_landmarks:
                        if info.flags.connection_on:
                            for num, hand in enumerate(results.multi_hand_landmarks):
                                mp_drawing.draw_landmarks(
                                    image, hand, mp_hands.HAND_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                    mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
                                )

                        landmarks_sn = SimpleNamespace(
                            timestamp=int(round(time.time() * 1000)),
                        )

                        preprocess(landmarks_sn, results, image, info)

                        detect_orientation(landmarks_sn, info)

                        detect_finger_self_occlusion(landmarks_sn, info)

                        detect_palm_occlusion(landmarks_sn, info)

                    else:
                        landmarks_sn = SimpleNamespace(
                            timestamp=int(round(time.time() * 1000)),
                            landmarks_list=[],
                            image=image
                        )

                    process_fingertip(landmarks_sn, info.blur_mode, kernel_size, normalized_kernel, info)

                    if info.flags.frame_rate_on:
                        curr_frame_time = time.time()
                        fps = int(1 / (curr_frame_time - prev_frame_time))
                        prev_frame_time = curr_frame_time
                        cv2.putText(
                            img=landmarks_sn.image,
                            text='{} FPS'.format(fps),
                            org=(10, 10),
                            fontFace=cv2.FONT_HERSHEY_PLAIN,
                            fontScale=1,
                            color=(0x00, 0xFF, 0x00),
                            thickness=2
                        )

                    processed_image = landmarks_sn.image
                    cv2.imwrite(
                        os.path.join(temp_path, '{}{}.jpeg'.format(0, landmarks_sn.timestamp)),
                        processed_image
                    )
                    processed_frame_count += 1

                    image_bytes = cv2.imencode('.png', processed_image)[1].tobytes()
                    window['image'].update(data=image_bytes)

                else:
                    _, frame = cap.read()
                    if option.flip:
                        frame = cv2.flip(frame, 1)
                    image_bytes = cv2.imencode('.png', frame)[1].tobytes()
                    window['image'].update(data=image_bytes)

        cap.release()
        cv2.destroyAllWindows()

    except KeyboardInterrupt:
        print('KeyboardInterrupt triggered by user in realtime process procedure.')

    finally:
        cap.release()
        cv2.destroyAllWindows()

    ret_dict['fingerprint_erase_end_timestamp'] = int(time.time() * 1000)
    window.close()
    finished_event.set()
    ret_dict['fingerprint_timestamp_list'] = timestamp_list
    ret_dict['frame_count'] = processed_frame_count


def sound_recorder(finished_event: Event, ret_dict: dict, info: types.SimpleNamespace) -> None:
    """Handling sound recording from microphone.

    Microphone is automatically selected by traversing available audio input sources.
    Args:
        finished_event (Event): tests the event if the stop is called in video recording and stops sound recording
        ret_dict (dict): audio record start timestamp and path to audio file
        info (types.SimpleNamespace): consts and attributes from main

    Returns:

    """
    device_no = 0
    sample_rate = 0
    channels = 2
    max_sample_rate_of_python_wav = 48000
    for device_no, device_dict in enumerate(list(sd.query_devices())):
        if device_dict['max_input_channels'] > 0:
            sample_rate = max(max_sample_rate_of_python_wav, int(device_dict['default_samplerate']))
            channels = min(channels, device_dict['max_input_channels'])
            break

    chunk = int(sample_rate / 2)
    sample_format = pyaudio.paInt16  # 16 bits per sample
    temp_path = info.folder
    filename = os.path.abspath(os.path.join(temp_path, "output.wav"))

    p = pyaudio.PyAudio()
    stream = p.open(
        format=sample_format,
        channels=channels,
        input_device_index=device_no,
        rate=sample_rate,
        frames_per_buffer=chunk,
        input=True
    )

    ret_dict['sound_recorder_start_timestamp'] = int(time.time() * 1000)
    frames = []  # Initialize list to store frames
    while not finished_event.is_set():
        data = stream.read(chunk, exception_on_overflow=False)
        frames.append(data)

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    p.terminate()

    # Save the recorded data as a WAV file
    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(sample_format))
    wf.setframerate(sample_rate)
    wf.writeframes(b''.join(frames))
    wf.close()
    ret_dict['record_path'] = filename


if __name__ == '__main__':
    ret = realtime_process()
    print(ret)
