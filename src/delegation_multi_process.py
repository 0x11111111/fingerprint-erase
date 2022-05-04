import json
import os
import time
from types import SimpleNamespace

import cv2
import mediapipe

from core_finger_processor import preprocess, detect_orientation, detect_finger_self_occlusion, detect_palm_occlusion, \
    process_fingertip


def multi_process_fingerprint_erase(group_number):
    prev_frame_time = 0
    curr_frame_time = 0

    with open('./info.json', 'r') as f:
        info = json.load(f, object_hook=lambda x: SimpleNamespace(**x))
        f.close()

    video_source = info.file_path
    temp_path = info.folder
    if not os.path.exists(temp_path):
        os.mkdir(temp_path)
    cap = cv2.VideoCapture(video_source)
    cap.set(cv2.CAP_PROP_POS_FRAMES, info.frame_jump_unit * group_number)
    processed_frame_count = 0

    mp_drawing = mediapipe.solutions.drawing_utils
    mp_hands = mediapipe.solutions.hands

    try:
        with mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.6) as hands:
            while processed_frame_count < info.frame_jump_unit:
                success, frame = cap.read()

                if not success:
                    break

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
                        timestamp=int(time.time() * 1000),
                    )

                    preprocess(landmarks_sn, results, image, info)

                    detect_orientation(landmarks_sn, info)

                    detect_finger_self_occlusion(landmarks_sn, info)

                    detect_palm_occlusion(landmarks_sn, info)

                else:
                    landmarks_sn = SimpleNamespace(
                        timestamp=int(time.time() * 1000),
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
                    os.path.join(temp_path, '{}{}.jpeg'.format(group_number, landmarks_sn.timestamp)),
                    processed_image
                )
                # cv2.imshow('Hand Tracking', processed_image)

                processed_frame_count += 1

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    interruption_flag = True
                    break

    except:
        cap.release()
        cv2.destroyAllWindows()

    cap.release()
    cv2.destroyAllWindows()
