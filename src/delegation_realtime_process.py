import os
import time
import json
import cv2
import mediapipe

import numpy as np
import PySimpleGUI as sg

from types import SimpleNamespace
from core_finger_processor import preprocess, detect_orientation, detect_finger_self_occlusion, detect_palm_occlusion, \
    process_fingertip


def realtime_process_fingerprint_erase(video_source=0):

    info = None
    if os.path.exists('./info.json'):
        with open('./info.json', 'r') as f:
            info = json.load(f, object_hook=lambda x: SimpleNamespace(**x))
            f.close()
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
            kernel_size=7,
            option=SimpleNamespace(
                averaging=True,
                gaussian=False,
                median=False,
                bilateral=False,
                nope=False
            ),
            blur_mode='averaging'
        )
        info.EPS = 0.0001
        info.tip_dip_length_ratio = 0.7
        info.pinky_ring_width_ratio = 0.89
        info.thumb_width_length_ratio = 0.62
        info.finger_mcp_width_ratio = 0.65
        info.landmark_order = 'abcdefghijklmnopqrstu'

    temp_path = info.folder
    if not os.path.exists(temp_path):
        os.mkdir(temp_path)

    cap = cv2.VideoCapture(video_source)

    mp_drawing = mediapipe.solutions.drawing_utils
    mp_hands = mediapipe.solutions.hands

    layout = [
        [sg.Image(filename='', key='image')],
        [
            sg.Button('Erase', size=(10, 1)),
            sg.Button('Stop', size=(10, 1)),
            sg.Button('Exit', size=(10, 1))
        ],
        [
            sg.Checkbox(key='flip', text='水平翻转'),
            sg.Checkbox(key='circle_on', text='核心指纹圈'),
            sg.Checkbox(key='connection_on', text='关键点连线'),
            sg.Checkbox(key='output_on', text='指纹核心参数'),
            sg.Checkbox(key='orientation_on', text='手掌朝向'),
            sg.Checkbox(key='frame_rate_on', text='帧率'),
            sg.Checkbox(key='box_on', text='手掌包络框'),
        ],
        [
            sg.Text('模糊程度'),
            sg.Slider(key='blur_value', range=(1, 51), default_value=info.kernel_size, size=(90, 15), orientation='horizontal')
        ],
        [
            sg.Text('模糊处理'),
            sg.Radio(key='averaging', text='平均平滑', group_id='blur_option', default=info.option.averaging),
            sg.Radio(key='gaussian', text='高斯模糊', group_id='blur_option', default=info.option.gaussian),
            sg.Radio(key='median', text='中值滤波', group_id='blur_option', default=info.option.median),
            sg.Radio(key='bilateral', text='双边滤波', group_id='blur_option', default=info.option.bilateral),
            sg.Radio(key='nope', text='不处理', group_id='blur_option', default=info.option.nope)
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

    try:
        with mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.6) as hands:
            while True:
                event, values = window.read(timeout=100)
                option = SimpleNamespace(**values)

                if event == 'Exit' or event == sg.WIN_CLOSED:
                    if erase_flag:
                        end_timestamp = int(time.time() * 1000)
                        timestamp_list.append((start_timestamp, end_timestamp))
                        erase_flag = False
                    break

                elif event == 'Stop':
                    if erase_flag:
                        end_timestamp = int(time.time() * 1000)
                        timestamp_list.append((start_timestamp, end_timestamp))
                        erase_flag = False

                elif event == 'Erase':
                    if not erase_flag:
                        start_timestamp = int(time.time() * 1000)
                        erase_flag = True

                if erase_flag:
                    ret, frame = cap.read()
                    if option.flip:
                        frame = cv2.flip(frame, 1)

                    info.flags.circle_on = option.circle_on
                    info.flags.connection_on = option.connection_on
                    info.flags.output_on = option.output_on
                    info.flags.orientation_on = option.orientation_on
                    info.flags.frame_rate_on = option.frame_rate_on
                    info.flags.box_on = option.box_on
                    info.kernel_size = int(option.blur_value // 2 * 2 + 1)
                    info.blur_mode = 'nope'
                    if option.averaging:
                        info.blur_mode = 'averaging'
                    elif option.gaussian:
                        info.blur_mode = 'gaussian'
                    elif option.median:
                        info.blur_mode = 'median'
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

                    process_fingertip(landmarks_sn, info.blur_mode, info.kernel_size, info)

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
                    ret, frame = cap.read()
                    if option.flip:
                        frame = cv2.flip(frame, 1)
                    image_bytes = cv2.imencode('.png', frame)[1].tobytes()
                    window['image'].update(data=image_bytes)

        cap.release()
        cv2.destroyAllWindows()

    except:
        cap.release()
        cv2.destroyAllWindows()

    window.close()
    return timestamp_list, processed_frame_count


if __name__ == '__main__':
    ret = realtime_process_fingerprint_erase(0)
    print(ret)
