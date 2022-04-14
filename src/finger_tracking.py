import mediapipe as mp
import cv2
import os
import time
import copy
import math
import numpy as np
from types import SimpleNamespace
from google.protobuf.json_format import MessageToDict

debug_mode = SimpleNamespace(track_on=True, coordination_on=True, output_on=False, orientation_on=True)

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)

if not os.path.exists("../.tmp"):
    os.mkdir("../.tmp")

folder = os.path.join('../.tmp', '{}'.format(int(round(time.time() * 1000))))
os.mkdir(folder)

EPS = 0.0001
finger_width_SimpleNamespace = SimpleNamespace(
    thumb=20.0,
    index=18.4,
    middle=18.3,
    ring=17.3,
    pinky=15.3
)
finger_length_SimpleNamespace = SimpleNamespace(
    thumb=32.1,
    index=24.7,
    middle=26.4,
    ring=26.3,
    pinky=23.7
)
tip_dip_length_ratio = 1.3

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
            if debug_mode.output_on:
                print(handedness_dict)

            for num, hand in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(
                    image, hand, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
                )

            hand_landmarks_SimpleNamespace = SimpleNamespace(
                timestamp=int(round(time.time() * 1000)),
                hand_landmarks_list=[],
            )

            # for idx, hand_world_landmarks in enumerate(results.multi_hand_world_landmarks):
            #     # print(hand_world_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x)
            #     # print(hand_world_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y)
            #     print(hand_world_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].z)

            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # print('hand_landmarks:', hand_landmarks)
                hand_landmarks_info = SimpleNamespace(
                    no=idx,
                    handedness=results.multi_handedness[idx].classification[0].label,
                    fingers=SimpleNamespace(
                        thumb=SimpleNamespace(
                            tip=SimpleNamespace(
                                x=hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * image_width,
                                y=hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * image_height,
                                z=hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].z,
                            ),
                            dip=SimpleNamespace(
                                x=hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x * image_width,
                                y=hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y * image_height,
                                z=hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].z
                            )
                        ),
                        index=SimpleNamespace(
                            tip=SimpleNamespace(
                                x=hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width,
                                y=hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height,
                                z=hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].z,
                            ),
                            dip=SimpleNamespace(
                                x=hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].x * image_width,
                                y=hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y * image_height,
                                z=hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].z
                            )
                        ),
                        middle=SimpleNamespace(
                            tip=SimpleNamespace(
                                x=hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x * image_width,
                                y=hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * image_height,
                                z=hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].z
                            ),
                            dip=SimpleNamespace(
                                x=hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].x * image_width,
                                y=hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y * image_height,
                                z=hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].z
                            )
                        ),
                        ring=SimpleNamespace(
                            tip=SimpleNamespace(
                                x=hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x * image_width,
                                y=hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y * image_height,
                                z=hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].z
                            ),
                            dip=SimpleNamespace(
                                x=hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].x * image_width,
                                y=hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].y * image_height,
                                z=hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].z
                            )
                        ),
                        pinky=SimpleNamespace(
                            tip=SimpleNamespace(
                                x=hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x * image_width,
                                y=hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y * image_height,
                                z=hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].z
                            ),
                            dip=SimpleNamespace(
                                x=hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].x * image_width,
                                y=hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].y * image_height,
                                z=hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].z
                            )
                        ),

                    ),
                    orientation_coord=SimpleNamespace(
                        wrist=SimpleNamespace(
                            x=hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * image_width,
                            y=hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * image_height,
                            z=hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].z
                        ),
                        thumb_cmc=SimpleNamespace(
                            x=hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].x * image_width,
                            y=hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].y * image_height,
                            z=hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].z
                        ),
                        pinky_mcp=SimpleNamespace(
                            x=hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].x * image_width,
                            y=hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y * image_height,
                            z=hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].z
                        )
                    )
                )

                for k, v in hand_landmarks_info.fingers.__dict__.items():
                    x_delta, y_delta = v.tip.x - v.dip.x, v.tip.y - v.dip.y
                    tip_dip_distance = math.sqrt(math.pow(x_delta, 2) + math.pow(y_delta, 2))
                    ratio = tip_dip_distance / tip_dip_length_ratio / finger_length_SimpleNamespace.__dict__[k]

                    if debug_mode.track_on:
                        cv2.circle(
                            img=image,
                            center=(int(v.tip.x), int(v.tip.y)),
                            radius=int(ratio * finger_width_SimpleNamespace.__dict__[k]),
                            color=(0x00, 0xFF, 0x00),
                            thickness=2
                        )

                wrist = np.array([hand_landmarks_info.orientation_coord.wrist.x,
                                  hand_landmarks_info.orientation_coord.wrist.y])
                thumb_cmc = np.array([hand_landmarks_info.orientation_coord.thumb_cmc.x,
                                      hand_landmarks_info.orientation_coord.thumb_cmc.y])
                pinky_mcp = np.array([hand_landmarks_info.orientation_coord.pinky_mcp.x,
                                      hand_landmarks_info.orientation_coord.pinky_mcp.y])

                orientation_angle = np.rad2deg(
                    np.arctan2(thumb_cmc[1] - wrist[1], thumb_cmc[0] - wrist[0]) -
                    np.arctan2(pinky_mcp[1] - wrist[1], pinky_mcp[0] - wrist[0])
                )
                if orientation_angle < -180:
                    orientation_angle += 360
                elif orientation_angle > 190:
                    orientation_angle -= 360

                if hand_landmarks_info.handedness == 'Left' and orientation_angle >= 0 \
                        or hand_landmarks_info.handedness == 'Right' and orientation_angle <= 0:
                    orientation = 'Front'
                else:
                    orientation = 'Rear'

                if debug_mode.orientation_on:
                    coord = (int(wrist[0]) + 10, int(wrist[1]))
                    text = 'orientation: {} angle: {}'.format(orientation, orientation_angle)
                    cv2.putText(
                        img=image,
                        text=text,
                        org=coord,
                        fontFace=cv2.FONT_HERSHEY_PLAIN,
                        fontScale=1,
                        color=(0x00, 0x00, 0xFF),
                        thickness=1
                    )

                if debug_mode.coordination_on:
                    text = f'handedness: {hand_landmarks_info.handedness}\n'
                    for k, v in hand_landmarks_info.fingers.__dict__.items():
                        text += \
                            '{} tip: ({:.3f}, {:.3f}, {:.6f})\n'.format(k, v.tip.x, v.tip.y, v.tip.z)

                    y0, dy = int(image_height * 0.1), 15
                    for i, line in enumerate(text.split('\n')):
                        cv2.putText(
                            img=image,
                            text=line,
                            org=(10, y0 + i * dy),
                            fontFace=cv2.FONT_HERSHEY_PLAIN,
                            fontScale=1,
                            color=(0x00, 0x00, 0xFF),
                            thickness=1
                        )

                hand_landmarks_SimpleNamespace.hand_landmarks_list.append(hand_landmarks_info)

            hand_landmarks_SimpleNamespace.image = copy.deepcopy(image)

            # print(text)

        else:
            hand_landmarks_SimpleNamespace = SimpleNamespace(
                timestamp=int(round(time.time() * 1000)),
                hand_landmarks_list=[],
                image=image
            )

        image_list.append(hand_landmarks_SimpleNamespace)
        cv2.imwrite(os.path.join(folder, '{}.jpeg'.format(hand_landmarks_SimpleNamespace.timestamp)), image)
        cv2.imshow('Hand Tracking', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
