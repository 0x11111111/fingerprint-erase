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
fingertip_radius_sn = SimpleNamespace(
    thumb=20.0,
    index=18.4,
    middle=18.3,
    ring=17.3,
    pinky=15.3
)
finger_length_sn = SimpleNamespace(
    thumb=32.1,
    index=24.7,
    middle=26.4,
    ring=26.3,
    pinky=23.7
)
tip_dip_length_ratio = 0.8
pinky_ring_width_ratio = 0.89
thumb_width_length_ratio = 0.60
finger_mcp_width_ratio = 0.8
image_list = []


def calculate_distance_sn(coord_sn_1, coord_sn_2):
    return math.sqrt((coord_sn_1.x - coord_sn_2.x) ** 2 + (coord_sn_1.y - coord_sn_2.y) ** 2)


def calculate_distance_array(coord_array_1, coord_array_2):
    return math.sqrt((coord_array_1[0] - coord_array_2[0]) ** 2 + (coord_array_1[1] - coord_array_2[1]) ** 2)


with mp_hands.Hands(min_detection_confidence=0.4, min_tracking_confidence=0.3) as hands:
    while cap.isOpened():
        success, frame = cap.read()

        if not success:
            print("Ignoring")
            continue

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.flip(image, 1)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

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

            y0, dy = int(image_height * 0.1), 15
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # print('hand_landmarks:', hand_landmarks)
                landmarks = SimpleNamespace(
                    no=idx,
                    handedness=results.multi_handedness[idx].classification[0].label,
                    wrist=SimpleNamespace(
                        x=hand_landmarks.landmark[0].x * image_width,
                        y=hand_landmarks.landmark[0].y * image_height,
                        z=hand_landmarks.landmark[0].z
                    ),

                    fingers=SimpleNamespace(
                        thumb=SimpleNamespace(
                            tip=SimpleNamespace(
                                x=hand_landmarks.landmark[4].x * image_width,
                                y=hand_landmarks.landmark[4].y * image_height,
                                z=hand_landmarks.landmark[4].z,
                            ),
                            ip=SimpleNamespace(
                                x=hand_landmarks.landmark[3].x * image_width,
                                y=hand_landmarks.landmark[3].y * image_height,
                                z=hand_landmarks.landmark[3].z
                            ),
                            mcp=SimpleNamespace(
                                x=hand_landmarks.landmark[2].x * image_width,
                                y=hand_landmarks.landmark[2].y * image_height,
                                z=hand_landmarks.landmark[2].z
                            ),
                            cmc=SimpleNamespace(
                                x=hand_landmarks.landmark[1].x * image_width,
                                y=hand_landmarks.landmark[1].y * image_height,
                                z=hand_landmarks.landmark[1].z
                            )
                        ),

                        index=SimpleNamespace(
                            tip=SimpleNamespace(
                                x=hand_landmarks.landmark[8].x * image_width,
                                y=hand_landmarks.landmark[8].y * image_height,
                                z=hand_landmarks.landmark[8].z,
                            ),
                            dip=SimpleNamespace(
                                x=hand_landmarks.landmark[7].x * image_width,
                                y=hand_landmarks.landmark[7].y * image_height,
                                z=hand_landmarks.landmark[7].z
                            ),
                            pip=SimpleNamespace(
                                x=hand_landmarks.landmark[6].x * image_width,
                                y=hand_landmarks.landmark[6].y * image_height,
                                z=hand_landmarks.landmark[6].z
                            ),
                            mcp=SimpleNamespace(
                                x=hand_landmarks.landmark[5].x * image_width,
                                y=hand_landmarks.landmark[5].y * image_height,
                                z=hand_landmarks.landmark[5].z
                            )
                        ),

                        middle=SimpleNamespace(
                            tip=SimpleNamespace(
                                x=hand_landmarks.landmark[12].x * image_width,
                                y=hand_landmarks.landmark[12].y * image_height,
                                z=hand_landmarks.landmark[12].z
                            ),
                            dip=SimpleNamespace(
                                x=hand_landmarks.landmark[11].x * image_width,
                                y=hand_landmarks.landmark[11].y * image_height,
                                z=hand_landmarks.landmark[11].z
                            ),
                            pip=SimpleNamespace(
                                x=hand_landmarks.landmark[10].x * image_width,
                                y=hand_landmarks.landmark[10].y * image_height,
                                z=hand_landmarks.landmark[10].z
                            ),
                            mcp=SimpleNamespace(
                                x=hand_landmarks.landmark[9].x * image_width,
                                y=hand_landmarks.landmark[9].y * image_height,
                                z=hand_landmarks.landmark[9].z
                            )
                        ),

                        ring=SimpleNamespace(
                            tip=SimpleNamespace(
                                x=hand_landmarks.landmark[16].x * image_width,
                                y=hand_landmarks.landmark[16].y * image_height,
                                z=hand_landmarks.landmark[16].z
                            ),
                            dip=SimpleNamespace(
                                x=hand_landmarks.landmark[15].x * image_width,
                                y=hand_landmarks.landmark[15].y * image_height,
                                z=hand_landmarks.landmark[15].z
                            ),
                            pip=SimpleNamespace(
                                x=hand_landmarks.landmark[14].x * image_width,
                                y=hand_landmarks.landmark[14].y * image_height,
                                z=hand_landmarks.landmark[14].z
                            ),
                            mcp=SimpleNamespace(
                                x=hand_landmarks.landmark[13].x * image_width,
                                y=hand_landmarks.landmark[13].y * image_height,
                                z=hand_landmarks.landmark[13].z
                            )
                        ),

                        pinky=SimpleNamespace(
                            tip=SimpleNamespace(
                                x=hand_landmarks.landmark[20].x * image_width,
                                y=hand_landmarks.landmark[20].y * image_height,
                                z=hand_landmarks.landmark[20].z
                            ),
                            dip=SimpleNamespace(
                                x=hand_landmarks.landmark[19].x * image_width,
                                y=hand_landmarks.landmark[19].y * image_height,
                                z=hand_landmarks.landmark[19].z
                            ),
                            pip=SimpleNamespace(
                                x=hand_landmarks.landmark[18].x * image_width,
                                y=hand_landmarks.landmark[18].y * image_height,
                                z=hand_landmarks.landmark[18].z
                            ),
                            mcp=SimpleNamespace(
                                x=hand_landmarks.landmark[17].x * image_width,
                                y=hand_landmarks.landmark[17].y * image_height,
                                z=hand_landmarks.landmark[17].z
                            )
                        ),

                    ),

                    finger_status=SimpleNamespace(
                        thumb=True,
                        index=True,
                        middle=True,
                        ring=True,
                        pinky=True
                    )
                )
                landmarks.mcp_width = SimpleNamespace(
                    index_middle=calculate_distance_sn(landmarks.fingers.index.mcp,
                                                       landmarks.fingers.middle.mcp),
                    middle_ring=calculate_distance_sn(landmarks.fingers.middle.mcp, landmarks.fingers.ring.mcp),
                    ring_pinky=calculate_distance_sn(landmarks.fingers.ring.mcp, landmarks.fingers.pinky.mcp)
                )

                landmarks.palm_width = landmarks.mcp_width.index_middle + landmarks.mcp_width.middle_ring + \
                    landmarks.mcp_width.ring_pinky

                landmarks.fingertip_major_axes = SimpleNamespace(
                    thumb=calculate_distance_sn(
                        landmarks.fingers.thumb.mcp,
                        landmarks.fingers.thumb.ip
                    ) * thumb_width_length_ratio * tip_dip_length_ratio * finger_mcp_width_ratio,
                    index=landmarks.mcp_width.index_middle / 2 * finger_mcp_width_ratio,
                    middle=landmarks.mcp_width.middle_ring / 2 * finger_mcp_width_ratio,
                    ring=landmarks.mcp_width.ring_pinky / 2 * finger_mcp_width_ratio,
                    pinky=landmarks.mcp_width.ring_pinky / 2 * pinky_ring_width_ratio * finger_mcp_width_ratio
                )

                wrist = np.array([landmarks.wrist.x,
                                  landmarks.wrist.y])
                thumb_cmc = np.array([landmarks.fingers.thumb.cmc.x,
                                      landmarks.fingers.thumb.cmc.y])
                pinky_mcp = np.array([landmarks.fingers.pinky.mcp.x,
                                      landmarks.fingers.pinky.mcp.y])

                thumb_cmc_wrist_deg = np.rad2deg(np.arctan2(thumb_cmc[1] - wrist[1], thumb_cmc[0] - wrist[0]))
                pinky_mcp_wrist_deg = np.rad2deg(np.arctan2(pinky_mcp[1] - wrist[1], pinky_mcp[0] - wrist[0]))
                orientation_angle = thumb_cmc_wrist_deg - pinky_mcp_wrist_deg
                if orientation_angle < -180:
                    orientation_angle += 360
                elif orientation_angle > 180:
                    orientation_angle -= 360

                if landmarks.handedness == 'Left' and orientation_angle >= 0 \
                        or landmarks.handedness == 'Right' and orientation_angle <= 0:
                    landmarks.orientation = 'Front'
                else:
                    landmarks.orientation = 'Rear'
                    landmarks.finger_status = SimpleNamespace(
                        thumb=False,
                        index=False,
                        middle=False,
                        ring=False,
                        pinky=False
                    )

                thumb_tip = np.array([landmarks.fingers.thumb.tip.x, landmarks.fingers.thumb.tip.y])
                thumb_ip = np.array([landmarks.fingers.thumb.ip.x, landmarks.fingers.thumb.ip.y])

                index_tip = np.array([landmarks.fingers.index.tip.x, landmarks.fingers.index.tip.y])
                index_dip = np.array([landmarks.fingers.index.dip.x, landmarks.fingers.index.dip.y])

                middle_tip = np.array([landmarks.fingers.middle.tip.x, landmarks.fingers.middle.tip.y])
                middle_dip = np.array([landmarks.fingers.middle.dip.x, landmarks.fingers.middle.dip.y])

                ring_tip = np.array([landmarks.fingers.ring.tip.x, landmarks.fingers.ring.tip.y])
                ring_dip = np.array([landmarks.fingers.ring.dip.x, landmarks.fingers.ring.dip.y])

                pinky_tip = np.array([landmarks.fingers.pinky.tip.x, landmarks.fingers.pinky.tip.y])
                pinky_dip = np.array([landmarks.fingers.pinky.dip.x, landmarks.fingers.pinky.dip.y])

                thumb_tip_ip_distance = calculate_distance_array(thumb_tip, thumb_ip)
                index_tip_dip_distance = calculate_distance_array(index_tip, index_dip)
                middle_tip_dip_distance = calculate_distance_array(middle_tip, middle_dip)
                ring_tip_dip_distance = calculate_distance_array(ring_tip, ring_dip)
                pinky_tip_dip_distance = calculate_distance_array(pinky_tip, pinky_dip)

                landmarks.fingertip_minor_axes = SimpleNamespace(
                    thumb=(1 + thumb_tip_ip_distance / finger_length_sn.thumb / tip_dip_length_ratio) * 0.4 * landmarks.fingertip_major_axes.thumb,
                    index=(1 + index_tip_dip_distance / finger_length_sn.index / tip_dip_length_ratio) * 0.4 * landmarks.fingertip_major_axes.index,
                    middle=(1 + middle_tip_dip_distance / finger_length_sn.middle / tip_dip_length_ratio) * 0.4 * landmarks.fingertip_major_axes.middle,
                    ring=(1 + ring_tip_dip_distance / finger_length_sn.ring / tip_dip_length_ratio) * 0.4 * landmarks.fingertip_major_axes.ring,
                    pinky=(1 + pinky_tip_dip_distance / finger_length_sn.pinky / tip_dip_length_ratio) * 0.4 * landmarks.fingertip_major_axes.pinky
                )

                landmarks.fingertip_angle = SimpleNamespace(
                    thumb=np.rad2deg(np.arctan2(thumb_tip[1] - thumb_ip[1], thumb_tip[0] - thumb_ip[0])) + 90,
                    index=np.rad2deg(np.arctan2(index_tip[1] - index_dip[1], index_tip[0] - index_dip[0])) + 90,
                    middle=np.rad2deg(np.arctan2(middle_tip[1] - middle_dip[1], middle_tip[0] - middle_dip[0])) + 90,
                    ring=np.rad2deg(np.arctan2(ring_tip[1] - ring_dip[1], ring_tip[0] - ring_dip[0])) + 90,
                    pinky=np.rad2deg(np.arctan2(pinky_tip[1] - pinky_dip[1], pinky_tip[0] - pinky_dip[0])) + 90
                )

                if landmarks.orientation == 'Front':
                    wrist_tip_distance = {
                        'thumb': calculate_distance_sn(landmarks.wrist, landmarks.fingers.thumb.tip),
                        'index': calculate_distance_sn(landmarks.wrist, landmarks.fingers.index.tip),
                        'middle': calculate_distance_sn(landmarks.wrist, landmarks.fingers.middle.tip),
                        'ring': calculate_distance_sn(landmarks.wrist, landmarks.fingers.ring.tip),
                        'pinky': calculate_distance_sn(landmarks.wrist, landmarks.fingers.pinky.tip)
                    }
                    wrist_cmp_distance = {
                        'thumb': calculate_distance_sn(landmarks.wrist, landmarks.fingers.thumb.ip),
                        'index': calculate_distance_sn(landmarks.wrist, landmarks.fingers.index.dip),
                        'middle': calculate_distance_sn(landmarks.wrist, landmarks.fingers.middle.dip),
                        'ring': calculate_distance_sn(landmarks.wrist, landmarks.fingers.ring.dip),
                        'pinky': calculate_distance_sn(landmarks.wrist, landmarks.fingers.pinky.dip)
                    }

                    for k, v in wrist_tip_distance.items():
                        if v < wrist_cmp_distance[k]:
                            landmarks.finger_status.__dict__[k] = False

                for k, v in landmarks.fingers.__dict__.items():
                    # radius = int(landmarks.fingertip_major_axes.__dict__[k])
                    center = (int(v.tip.x), int(v.tip.y))
                    major_axe = int(landmarks.fingertip_major_axes.__dict__[k])
                    minor_axe = int(landmarks.fingertip_minor_axes.__dict__[k])
                    angle = landmarks.fingertip_angle.__dict__[k]

                    if debug_mode.track_on:
                        text = 'Major: {}, minor: {}, angle: {:.3f}'.format(major_axe, minor_axe, angle)
                        cv2.putText(
                            img=image,
                            text=text,
                            org=(center[0] + 10, center[1] + 10),
                            fontFace=cv2.FONT_HERSHEY_PLAIN,
                            fontScale=1,
                            color=(0x00, 0x00, 0xFF),
                            thickness=1
                        )

                        if landmarks.finger_status.__dict__[k]:
                            cv2.ellipse(
                                img=image,
                                center=center,
                                axes=(major_axe, minor_axe),
                                angle=angle,
                                startAngle=0,
                                endAngle=360,
                                color=(0x00, 0xFF, 0x00),
                                thickness=2
                            )
                        else:
                            cv2.ellipse(
                                img=image,
                                center=center,
                                axes=(major_axe, minor_axe),
                                angle=angle,
                                startAngle=0,
                                endAngle=360,
                                color=(0x00, 0x00, 0xFF),
                                thickness=2
                            )

                    # if debug_mode.track_on:
                    #     if landmarks.finger_status.__dict__[k]:
                    #         cv2.circle(
                    #             img=image,
                    #             center=center,
                    #             radius=radius,
                    #             color=(0x00, 0xFF, 0x00),
                    #             thickness=2
                    #         )
                    #     else:
                    #         cv2.circle(
                    #             img=image,
                    #             center=center,
                    #             radius=radius,
                    #             color=(0x00, 0x00, 0xFF),
                    #             thickness=2
                    #         )

                if debug_mode.orientation_on:
                    coord1 = (int(wrist[0]) + 10, int(wrist[1]))
                    text = 'orientation: {} angle: {}'.format(
                        landmarks.orientation, orientation_angle
                    )
                    cv2.putText(
                        img=image,
                        text=text,
                        org=coord1,
                        fontFace=cv2.FONT_HERSHEY_PLAIN,
                        fontScale=1,
                        color=(0x00, 0x00, 0xFF),
                        thickness=1
                    )

                    coord2 = (int(wrist[0]) + 10, int(wrist[1]) + 10)
                    text = 't_w: {} p_w: {}'.format(
                        thumb_cmc_wrist_deg, pinky_mcp_wrist_deg
                    )
                    cv2.putText(
                        img=image,
                        text=text,
                        org=coord2,
                        fontFace=cv2.FONT_HERSHEY_PLAIN,
                        fontScale=1,
                        color=(0x00, 0x00, 0xFF),
                        thickness=1
                    )

                if debug_mode.coordination_on:
                    text = f'handedness: {landmarks.handedness}\n'
                    for k, v in landmarks.fingers.__dict__.items():
                        text += \
                            '{} tip: ({:.3f}, {:.3f}, {:.6f})\n'.format(k, v.tip.x, v.tip.y, v.tip.z)

                    for i, line in enumerate(text.split('\n')):
                        cv2.putText(
                            img=image,
                            text=line,
                            org=(10, y0),
                            fontFace=cv2.FONT_HERSHEY_PLAIN,
                            fontScale=1,
                            color=(0x00, 0x00, 0xFF),
                            thickness=1
                        )
                        y0 += dy

                hand_landmarks_SimpleNamespace.hand_landmarks_list.append(landmarks)

            hand_landmarks_SimpleNamespace.image = copy.deepcopy(image)

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
