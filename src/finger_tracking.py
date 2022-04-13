import mediapipe as mp
import cv2
import numpy as np
import uuid
import os
import time
import copy
import math
from types import SimpleNamespace as map
from google.protobuf.json_format import MessageToDict

debug_mode = map(text_on=True)

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)

if not os.path.exists("../.tmp"):
    os.mkdir("../.tmp")

folder = os.path.join('../.tmp', '{}'.format(int(round(time.time() * 1000))))
os.mkdir(folder)

EPS = 0.0001
finger_width_map = map(
    thumb=20.0,
    index=18.4,
    middle=18.3,
    ring=17.3,
    pinky=15.3
)
finger_length_map = map(
    thumb=32.1,
    index=24.7,
    middle=26.4,
    ring=26.3,
    pinky=23.7
)
tip_dip_length_ratio = 0.75

image_list = []

with mp_hands.Hands(min_detection_confidence=0.4, min_tracking_confidence=0.3) as hands:
    while cap.isOpened():
        success, frame = cap.read()

        if not success:
            print("Ignoring")
            continue

        # BGR 2 RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Flip on horizontal
        image = cv2.flip(image, 1)

        # Set flag
        image.flags.writeable = False

        # Detections
        results = hands.process(image)

        # Set flag to true
        image.flags.writeable = True

        # RGB 2 BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Detections
        # print(results.multi_hand_landmarks)

        image_height, image_width, _ = image.shape

        text = ''
        # Rendering results
        if results.multi_hand_landmarks:
            for idx, handedness in enumerate(results.multi_handedness):
                handedness_dict = MessageToDict(handedness)
            print(handedness_dict)

            for num, hand in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(
                    image, hand, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
                )

            hand_landmarks_map = map(
                timestamp=int(round(time.time() * 1000)),
                hand_landmarks_list=[],
            )

            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # print('hand_landmarks:', hand_landmarks)
                hand_landmarks_info = map(
                    no=idx,
                    handedness=results.multi_handedness[idx].classification[0].label,
                    fingers=map(
                        thumb=map(
                            tip=map(
                                x=hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * image_width,
                                y=hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * image_height,
                                z=hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].z,
                            ),
                            dip=map(
                                x=hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x * image_width,
                                y=hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y * image_height,
                                z=hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].z
                            )
                        ),
                        index=map(
                            tip=map(
                                x=hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width,
                                y=hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height,
                                z=hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].z,
                            ),
                            dip=map(
                                x=hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].x * image_width,
                                y=hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y * image_height,
                                z=hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].z
                            )
                        ),
                        middle=map(
                            tip=map(
                                x=hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x * image_width,
                                y=hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * image_height,
                                z=hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].z
                            ),
                            dip=map(
                                x=hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].x * image_width,
                                y=hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y * image_height,
                                z=hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].z
                            )
                        ),
                        ring=map(
                            tip=map(
                                x=hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x * image_width,
                                y=hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y * image_height,
                                z=hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].z
                            ),
                            dip=map(
                                x=hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].x * image_width,
                                y=hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].y * image_height,
                                z=hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].z
                            )
                        ),
                        pinky=map(
                            tip=map(
                                x=hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x * image_width,
                                y=hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y * image_height,
                                z=hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].z
                            ),
                            dip=map(
                                x=hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].x * image_width,
                                y=hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].y * image_height,
                                z=hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].z
                            )
                        ),
                    )
                )

                # hand_landmarks_info.length_distance_ratio = ratio = math.fabs(
                #     hand_landmarks_info.fingers.middle.tip.y - hand_landmarks_info.fingers.middle.dip.y
                # ) / (finger_length_map.middle * tip_dip_length_ratio)
                y0 = 10
                for k, v in hand_landmarks_info.fingers.__dict__.items():
                    x_delta, y_delta = v.tip.x - v.dip.x, v.tip.y - v.dip.y
                    tip_dip_distance = math.sqrt(math.pow(x_delta, 2) + math.pow(y_delta, 2))
                    vertical_distance = tip_dip_distance / tip_dip_length_ratio
                    ratio = tip_dip_distance / tip_dip_length_ratio / finger_length_map.__dict__[k]
                    horizontal_distance = finger_width_map.__dict__[k] * ratio

                    cv2.putText(
                        img=image,
                        text="{}: x_delta: {:.2f}, y_delta: {:.2f}".format(k, x_delta, y_delta),
                        org=(10, y0),
                        fontFace=cv2.FONT_HERSHEY_PLAIN,
                        fontScale=1,
                        color=(0x00, 0x00, 0xFF),
                        thickness=1
                    )
                    y0 += 20
                    if -EPS < x_delta < EPS:
                        v.left_down_pos = map(
                            x=v.dip.x,
                            y=v.dip.y - horizontal_distance / 2
                        )
                        v.right_down_pos = map(
                            x=v.dip.x,
                            y=v.dip.y + horizontal_distance / 2
                        )
                        v.left_up_pos = map(
                            x=v.dip.x + vertical_distance,
                            y=v.dip.y - horizontal_distance / 2
                        )
                        v.right_up_pos = map(
                            x=v.dip.x + vertical_distance,
                            y=v.dip.y + horizontal_distance / 2
                        )
                    else:
                        angle = math.atan(y_delta / x_delta)
                        sine = math.sin(angle)
                        cosine = math.cos(angle)
                        cv2.putText(
                            img=image,
                            text="angle: {:.2f}, sin: {:.2f}, cos: {:.2f}".format(angle, sine, cosine),
                            org=(10, y0),
                            fontFace=cv2.FONT_HERSHEY_PLAIN,
                            fontScale=1,
                            color=(0x00, 0x00, 0xFF),
                            thickness=1
                        )
                        y0 += 20
                        v.left_down_pos = map(
                            x=v.dip.x - horizontal_distance / 2 * sine,
                            y=v.dip.y + horizontal_distance / 2 * cosine
                        )
                        v.right_down_pos = map(
                            x=v.dip.x + horizontal_distance / 2 * sine,
                            y=v.dip.y - horizontal_distance / 2 * cosine
                        )
                        v.left_up_pos = map(
                            x=v.left_down_pos.x + vertical_distance * cosine,
                            y=v.left_down_pos.y + vertical_distance * sine
                        )
                        v.right_up_pos = map(
                            x=v.right_down_pos.x + vertical_distance * cosine,
                            y=v.right_down_pos.y + vertical_distance * sine
                        )

                    cv2.line(
                        img=image,
                        pt1=(int(v.left_down_pos.x), int(v.left_down_pos.y)),
                        pt2=(int(v.right_down_pos.x), int(v.right_down_pos.y)),
                        color=(0x00, 0xFF, 0x00),
                        thickness=2
                    )
                    cv2.line(
                        img=image,
                        pt1=(int(v.right_down_pos.x), int(v.right_down_pos.y)),
                        pt2=(int(v.right_up_pos.x), int(v.right_up_pos.y)),
                        color=(0x00, 0xFF, 0x00),
                        thickness=2
                    )
                    cv2.line(
                        img=image,
                        pt1=(int(v.left_up_pos.x), int(v.left_up_pos.y)),
                        pt2=(int(v.right_up_pos.x), int(v.right_up_pos.y)),
                        color=(0x00, 0xFF, 0x00),
                        thickness=2
                    )
                    cv2.line(
                        img=image,
                        pt1=(int(v.left_up_pos.x), int(v.left_up_pos.y)),
                        pt2=(int(v.left_down_pos.x), int(v.left_down_pos.y)),
                        color=(0x00, 0xFF, 0x00),
                        thickness=2
                    )

                hand_landmarks_map.hand_landmarks_list.append(hand_landmarks_info)
            hand_landmarks_map.image = copy.deepcopy(image)


                # text += f'HANDEDNESS: {results.multi_handedness[idx].classification[0].label}\n' + \
                #     f'THUMB tip coordinates: (' + \
                #     f'{hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * image_width}, ' +\
                #     f'{hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * image_height})\n' +\
                #     f'INDEX finger tip coordinates: (' +\
                #     f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, ' +\
                #     f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})\n' +\
                #     f'MIDDLE finger tip coordinates: (' +\
                #     f'{hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x * image_width}, ' +\
                #     f'{hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * image_height})\n' +\
                #     f'RING finger tip coordinates: (' +\
                #     f'{hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x * image_width}, ' +\
                #     f'{hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y * image_height})\n' +\
                #     f'PINKY tip coordinates: (' +\
                #     f'{hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x * image_width}, ' +\
                #     f'{hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y * image_height})\n'

            # y0, dy = int(image_height * 0.1), 15
            # for i, line in enumerate(text.split('\n')):
            #     cv2.putText(
            #         img=image,
            #         text=line,
            #         org=(10, y0 + i * dy),
            #         fontFace=cv2.FONT_HERSHEY_PLAIN,
            #         fontScale=0.8,
            #         color=(0x00, 0x00, 0xFF),
            #         thickness=1
            #     )

            # print(text)

        else:
            hand_landmarks_map = map(
                timestamp=int(round(time.time() * 1000)),
                hand_landmarks_list=[],
                image=image
            )

        image_list.append(hand_landmarks_map)
        # print(image_list[-1])
        cv2.imwrite(os.path.join(folder, '{}.jpeg'.format(int(round(time.time() * 1000)))), image)
        cv2.imshow('Hand Tracking', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
