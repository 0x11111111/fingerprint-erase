import mediapipe as mp
import cv2
import os
import time
import copy
import math
import numpy as np
from types import SimpleNamespace
from google.protobuf.json_format import MessageToDict

debug_mode = SimpleNamespace(
    track_on=True,
    coordination_on=False,
    output_on=False,
    orientation_on=True,
    frame_rate_on=True,
    scoop_on=True
)

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

video_source = '../test/5.mp4'
# video_source = 0
cap = cv2.VideoCapture(video_source)

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
finger_mcp_width_ratio = 0.65
prev_frame_time = 0
curr_frame_time = 0
landmark_order = 'abcdefghijklmnopqrstu'
image_list = []


def calculate_distance_sn(coord_sn_1, coord_sn_2):
    return math.sqrt((coord_sn_1.x - coord_sn_2.x) ** 2 + (coord_sn_1.y - coord_sn_2.y) ** 2)


def calculate_distance_array(coord_array_1, coord_array_2):
    return math.sqrt((coord_array_1[0] - coord_array_2[0]) ** 2 + (coord_array_1[1] - coord_array_2[1]) ** 2)


# def intersect_line_circle(p, lsp, lep):
#     # p is the circle parameter, lsp and lep is the two end of the line
#     px, py, rr = p
#     px1, py1 = lsp
#     px2, py2 = lep
#
#     if px1 == px2:
#         if abs(rr) >= abs(px1 - px):
#             return True
#         else:
#             return False
#
#     else:
#         slope = (py1 - py2) / (px1 - px2)
#         b0 = py1 - slope * px1
#         a = slope ** 2 + 1
#         b = 2 * slope * (b0 - py) - 2 * px
#         c = (b0 - py) ** 2 + px ** 2 - rr ** 2
#         delta = b ** 2 - 4 * a * c
#         if delta >= 0:
#             return True
#         else:
#             return False

def intersect_line_circle(circle_center, circle_radius, pt1, pt2, full_line=False, tangent_tol=1e-9):
    """ Find the points at which a circle intersects a line-segment.  This can happen at 0, 1, or 2 points.

    :param circle_center: The (x, y) location of the circle center
    :param circle_radius: The radius of the circle
    :param pt1: The (x, y) location of the first point of the segment
    :param pt2: The (x, y) location of the second point of the segment
    :param full_line: True to find intersections along full line - not just in the segment.  False will just return intersections within the segment.
    :param tangent_tol: Numerical tolerance at which we decide the intersections are close enough to consider it a tangent
    :return Sequence[Tuple[float, float]]: A list of length 0, 1, or 2, where each element is a point at which the circle intercepts a line segment.

    """

    (p1x, p1y), (p2x, p2y), (cx, cy) = pt1, pt2, circle_center
    (x1, y1), (x2, y2) = (p1x - cx, p1y - cy), (p2x - cx, p2y - cy)
    dx, dy = (x2 - x1), (y2 - y1)
    dr = (dx ** 2 + dy ** 2) ** .5
    big_d = x1 * y2 - x2 * y1
    discriminant = circle_radius ** 2 * dr ** 2 - big_d ** 2

    if discriminant < 0:  # No intersection between circle and line
        return []
    else:  # There may be 0, 1, or 2 intersections with the segment
        intersections = [
            (cx + (big_d * dy + sign * (-1 if dy < 0 else 1) * dx * discriminant ** .5) / dr ** 2,
             cy + (-big_d * dx + sign * abs(dy) * discriminant ** .5) / dr ** 2)
            for sign in ((1, -1) if dy < 0 else (-1, 1))
        ]  # This makes sure the order along the segment is correct
        if not full_line:  # If only considering the segment, filter out intersections that do not fall within the segment
            fraction_along_segment = [(xi - p1x) / dx if abs(dx) > abs(dy) else (yi - p1y) / dy for xi, yi in
                                      intersections]
            intersections = [pt for pt, frac in zip(intersections, fraction_along_segment) if 0 <= frac <= 1]
        if len(intersections) == 2 and abs(
                discriminant) <= tangent_tol:  # If line is tangent to circle, return just one point (as both intersections have same location)
            return [intersections[0]]
        else:
            return intersections


with mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.6) as hands:
    while cap.isOpened():
        success, frame = cap.read()

        if not success:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if isinstance(video_source, int):
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

            landmarks_sn = SimpleNamespace(
                timestamp=int(round(time.time() * 1000)),
                landmarks_list=[],
            )

            yy, dy = int(image_height * 0.1), 15
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

                landmarks.landmark = SimpleNamespace(
                    a=landmarks.wrist,

                    b=landmarks.fingers.thumb.cmc,
                    c=landmarks.fingers.thumb.mcp,
                    d=landmarks.fingers.thumb.ip,
                    e=landmarks.fingers.thumb.tip,

                    f=landmarks.fingers.index.mcp,
                    g=landmarks.fingers.index.pip,
                    h=landmarks.fingers.index.dip,
                    i=landmarks.fingers.index.tip,

                    j=landmarks.fingers.middle.mcp,
                    k=landmarks.fingers.middle.pip,
                    l=landmarks.fingers.middle.dip,
                    m=landmarks.fingers.middle.tip,

                    n=landmarks.fingers.ring.mcp,
                    o=landmarks.fingers.ring.pip,
                    p=landmarks.fingers.ring.dip,
                    q=landmarks.fingers.ring.tip,

                    r=landmarks.fingers.pinky.mcp,
                    s=landmarks.fingers.pinky.pip,
                    t=landmarks.fingers.pinky.dip,
                    u=landmarks.fingers.pinky.tip
                )

                # Assign an alias for thumb.ip
                landmarks.fingers.thumb.dip = SimpleNamespace(
                    x=landmarks.fingers.thumb.ip.x,
                    y=landmarks.fingers.thumb.ip.y,
                    z=landmarks.fingers.thumb.ip.z
                )

                landmarks.mcp_width = SimpleNamespace(
                    index_middle=calculate_distance_sn(landmarks.fingers.index.mcp,
                                                       landmarks.fingers.middle.mcp),
                    middle_ring=calculate_distance_sn(landmarks.fingers.middle.mcp, landmarks.fingers.ring.mcp),
                    ring_pinky=calculate_distance_sn(landmarks.fingers.ring.mcp, landmarks.fingers.pinky.mcp)
                )

                landmarks.fingertip_distance_aggregated = landmarks.mcp_width.index_middle + \
                    landmarks.mcp_width.middle_ring + landmarks.mcp_width.ring_pinky

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

                wrist = np.array([landmarks.wrist.x, landmarks.wrist.y])
                thumb_cmc = np.array([landmarks.fingers.thumb.cmc.x, landmarks.fingers.thumb.cmc.y])
                pinky_mcp = np.array([landmarks.fingers.pinky.mcp.x, landmarks.fingers.pinky.mcp.y])

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
                    thumb=(landmarks.fingertip_major_axes.thumb + thumb_tip_ip_distance) * 0.5,
                    index=(landmarks.fingertip_major_axes.index + index_tip_dip_distance) * 0.5,
                    middle=(landmarks.fingertip_major_axes.middle + middle_tip_dip_distance) * 0.5,
                    ring=(landmarks.fingertip_major_axes.ring + ring_tip_dip_distance) * 0.5,
                    pinky=(landmarks.fingertip_major_axes.pinky + pinky_tip_dip_distance) * 0.5
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

                    coord2 = (int(wrist[0]) + 10, int(wrist[1]) + 15)
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
                        text += '{} tip: ({:.3f}, {:.3f}, {:.6f})\n'.format(k, v.tip.x, v.tip.y, v.tip.z)

                    for i, line in enumerate(text.split('\n')):
                        cv2.putText(
                            img=image,
                            text=line,
                            org=(10, yy),
                            fontFace=cv2.FONT_HERSHEY_PLAIN,
                            fontScale=1,
                            color=(0x00, 0x00, 0xFF),
                            thickness=1
                        )
                        yy += dy

                landmarks_sn.landmarks_list.append(landmarks)

            landmarks_sn.image = copy.deepcopy(image)

        else:
            landmarks_sn = SimpleNamespace(
                timestamp=int(round(time.time() * 1000)),
                landmarks_list=[],
                image=image
            )

        if landmarks_sn.landmarks_list:
            # hand occlusion detection
            ls = landmarks_sn.landmarks_list
            if len(ls) > 1:
                if ls[0].fingertip_distance_aggregated > ls[1].fingertip_distance_aggregated:
                    close, distant = ls[0], ls[1]
                else:
                    close, distant = ls[1], ls[0]

                # Avoid detection out of frame
                close_x0 = distant_x0 = image_width * 2
                close_y0 = distant_y0 = image_height * 2
                close_x1 = distant_x1 = -image_width
                close_y1 = distant_y1 = -image_height
                for i in landmark_order:
                    if close.landmark.__dict__[i].x < close_x0:
                        close_x0 = close.landmark.__dict__[i].x
                    if close.landmark.__dict__[i].x > close_x1:
                        close_x1 = close.landmark.__dict__[i].x
                    if close.landmark.__dict__[i].y < close_y0:
                        close_y0 = close.landmark.__dict__[i].y
                    if close.landmark.__dict__[i].y > close_y1:
                        close_y1 = close.landmark.__dict__[i].y

                    if distant.landmark.__dict__[i].x < distant_x0:
                        distant_x0 = distant.landmark.__dict__[i].x
                    if distant.landmark.__dict__[i].x > distant_x1:
                        distant_x1 = distant.landmark.__dict__[i].x
                    if distant.landmark.__dict__[i].y < distant_y0:
                        distant_y0 = distant.landmark.__dict__[i].y
                    if distant.landmark.__dict__[i].y > distant_y1:
                        distant_y1 = distant.landmark.__dict__[i].y

                if debug_mode.scoop_on:
                    cv2.putText(
                        img=image,
                        text='Close',
                        org=(int(close_x0) + 10, int(close_y0) + 10),
                        fontFace=cv2.FONT_HERSHEY_PLAIN,
                        fontScale=1,
                        color=(0x00, 0x00, 0xFF),
                        thickness=1
                    )
                    cv2.rectangle(
                        img=image,
                        pt1=(int(close_x0), int(close_y0)),
                        pt2=(int(close_x1), int(close_y1)),
                        color=(0x00, 0xFF, 0x00)
                    )

                    cv2.putText(
                        img=image,
                        text='Distant',
                        org=(int(distant_x0) + 10, int(distant_y0) + 10),
                        fontFace=cv2.FONT_HERSHEY_PLAIN,
                        fontScale=1,
                        color=(0x00, 0x00, 0xFF),
                        thickness=1
                    )
                    cv2.rectangle(
                        img=image,
                        pt1=(int(distant_x0), int(distant_y0)),
                        pt2=(int(distant_x1), int(distant_y1)),
                        color=(0x00, 0xFF, 0x00)
                    )

                if not (close_x0 > distant_x1 or close_x1 < distant_x0
                        or close_y0 > distant_y1 or close_y1 < distant_y0):

                    connections_number = set(mp_hands.HAND_CONNECTIONS)
                    connections_number.add((0, 9))
                    connections_number.add((0, 13))
                    connections_number.add((1, 5))
                    connections_number.add((2, 5))

                    for ki, vi in distant.fingers.__dict__.items():
                        k_coord = np.array([int(vi.tip.x * 0.7 + vi.dip.x * 0.3), int(vi.tip.y * 0.7 + vi.dip.y * 0.3)])
                        xi, yi = int(vi.tip.x * 0.7 + vi.dip.x * 0.3), int(vi.tip.y * 0.7 + vi.dip.y * 0.3)
                        ri = int(distant.fingertip_major_axes.__dict__[ki] * 2.5)

                        # for kj, vj in close.landmark.__dict__.items():
                        #     if ri >= calculate_distance_array(k_coord, np.array([int(vj.x), int(vj.y)])):
                        #         print('occlusion')
                        #         distant.finger_status.__dict__[ki] = False

                        for first, second in connections_number:
                            x1 = int(close.landmark.__dict__[landmark_order[first]].x)
                            y1 = int(close.landmark.__dict__[landmark_order[first]].y)
                            x2 = int(close.landmark.__dict__[landmark_order[second]].x)
                            y2 = int(close.landmark.__dict__[landmark_order[second]].y)
                            intersect_res = intersect_line_circle((xi, yi), ri, (x1, y1), (x2, y2))
                            if intersect_res:
                                distant.finger_status.__dict__[ki] = False

        for landmark in landmarks_sn.landmarks_list:
            for k, v in landmark.fingers.__dict__.items():
                center = (int(v.tip.x * 0.7 + v.dip.x * 0.3), int(v.tip.y * 0.7 + v.dip.y * 0.3))
                major_axe = int(landmark.fingertip_major_axes.__dict__[k])
                minor_axe = int(landmark.fingertip_minor_axes.__dict__[k])
                angle = landmark.fingertip_angle.__dict__[k]

                if debug_mode.track_on:
                    # text = 'Major: {}, minor: {}, angle: {:.3f}'.format(major_axe, minor_axe, angle)
                    # cv2.putText(
                    #     img=image,
                    #     text=text,
                    #     org=(center[0] + 10, center[1] + 10),
                    #     fontFace=cv2.FONT_HERSHEY_PLAIN,
                    #     fontScale=1,
                    #     color=(0x00, 0x00, 0xFF),
                    #     thickness=1
                    # )

                    if landmark.finger_status.__dict__[k]:
                        cv2.ellipse(
                            img=landmarks_sn.image,
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
                            img=landmarks_sn.image,
                            center=center,
                            axes=(major_axe, minor_axe),
                            angle=angle,
                            startAngle=0,
                            endAngle=360,
                            color=(0x00, 0x00, 0xFF),
                            thickness=2
                        )

        if debug_mode.frame_rate_on:
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
                thickness=1
            )

        image_list.append(landmarks_sn)
        cv2.imwrite(os.path.join(folder, '{}.jpeg'.format(landmarks_sn.timestamp)), landmarks_sn.image)

        cv2.imshow('Hand Tracking', landmarks_sn.image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
