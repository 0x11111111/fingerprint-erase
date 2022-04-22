import mediapipe as mp
import cv2
import os
import time
import math
import numpy as np
from types import SimpleNamespace

debug_mode = SimpleNamespace(
    track_on=False,
    landmark_on=False,
    coordination_on=False,
    output_on=False,
    orientation_on=False,
    frame_rate_on=True,
    scoop_on=False,
    blur_on=2
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


def intersect_line_circle(circle_center, circle_radius, pt1, pt2, full_line=False, tangent_tol=1e-4):
    """ Find the points at which a circle intersects a line-segment.  This can happen at 0, 1, or 2 points.

    param circle_center: The (x, y) location of the circle center
    param circle_radius: The radius of the circle
    param pt1: The (x, y) location of the first point of the segment
    param pt2: The (x, y) location of the second point of the segment
    param full_line: True to find intersections along full line - not just in the segment.  False will just
        return intersections within the segment.
    param tangent_tol: Numerical tolerance at which we decide the intersections are close enough to
        consider it a tangent
    return Sequence[Tuple[float, float]]: A list of length 0, 1, or 2, where each element is a point at
        which the circle intercepts a line segment.

    """

    (p1x, p1y), (p2x, p2y), (cx, cy) = pt1, pt2, circle_center
    (x1, y1), (x2, y2) = (p1x - cx, p1y - cy), (p2x - cx, p2y - cy)
    dx, dy = (x2 - x1), (y2 - y1)
    dr = (dx ** 2 + dy ** 2) ** 0.5
    big_d = x1 * y2 - x2 * y1
    discriminant = circle_radius ** 2 * dr ** 2 - big_d ** 2

    if discriminant < 0:  # No intersection between circle and line
        return []
    else:  # There may be 0, 1, or 2 intersections with the segment
        intersections = [
            (cx + (big_d * dy + sign * (-1 if dy < 0 else 1) * dx * discriminant ** 0.5) / dr ** 2,
             cy + (-big_d * dx + sign * abs(dy) * discriminant ** 0.5) / dr ** 2)
            for sign in ((1, -1) if dy < 0 else (-1, 1))
        ]  # This makes sure the order along the segment is correct
        if not full_line:
            # If only considering the segment, filter out intersections that do not fall within the segment
            fraction_along_segment = [(xi - p1x) / dx if abs(dx) > abs(dy) else (yi - p1y) / dy for xi, yi in
                                      intersections]
            intersections = [pt for pt, frac in zip(intersections, fraction_along_segment) if 0 <= frac <= 1]
        if len(intersections) == 2 and abs(
                discriminant) <= tangent_tol:
            # If line is tangent to circle, return just one point (as both intersections have same location)
            return [intersections[0]]
        else:
            return intersections


def preprocess(_landmarks_sn, res, _image):
    height, width, _ = _image.shape
    _landmarks_sn.landmarks_list = []
    _yy, _dy = 10, 15
    for idx, hand_landmarks in enumerate(res.multi_hand_landmarks):
        # print('hand_landmarks:', hand_landmarks)
        _landmarks = SimpleNamespace(
            no=idx,
            handedness=res.multi_handedness[idx].classification[0].label,
            wrist=SimpleNamespace(
                x=hand_landmarks.landmark[0].x * width,
                y=hand_landmarks.landmark[0].y * height,
                z=hand_landmarks.landmark[0].z
            ),

            fingers=SimpleNamespace(
                thumb=SimpleNamespace(
                    tip=SimpleNamespace(
                        x=hand_landmarks.landmark[4].x * width,
                        y=hand_landmarks.landmark[4].y * height,
                        z=hand_landmarks.landmark[4].z,
                    ),
                    ip=SimpleNamespace(
                        x=hand_landmarks.landmark[3].x * width,
                        y=hand_landmarks.landmark[3].y * height,
                        z=hand_landmarks.landmark[3].z
                    ),
                    mcp=SimpleNamespace(
                        x=hand_landmarks.landmark[2].x * width,
                        y=hand_landmarks.landmark[2].y * height,
                        z=hand_landmarks.landmark[2].z
                    ),
                    cmc=SimpleNamespace(
                        x=hand_landmarks.landmark[1].x * width,
                        y=hand_landmarks.landmark[1].y * height,
                        z=hand_landmarks.landmark[1].z
                    )
                ),

                index=SimpleNamespace(
                    tip=SimpleNamespace(
                        x=hand_landmarks.landmark[8].x * width,
                        y=hand_landmarks.landmark[8].y * height,
                        z=hand_landmarks.landmark[8].z,
                    ),
                    dip=SimpleNamespace(
                        x=hand_landmarks.landmark[7].x * width,
                        y=hand_landmarks.landmark[7].y * height,
                        z=hand_landmarks.landmark[7].z
                    ),
                    pip=SimpleNamespace(
                        x=hand_landmarks.landmark[6].x * width,
                        y=hand_landmarks.landmark[6].y * height,
                        z=hand_landmarks.landmark[6].z
                    ),
                    mcp=SimpleNamespace(
                        x=hand_landmarks.landmark[5].x * width,
                        y=hand_landmarks.landmark[5].y * height,
                        z=hand_landmarks.landmark[5].z
                    )
                ),

                middle=SimpleNamespace(
                    tip=SimpleNamespace(
                        x=hand_landmarks.landmark[12].x * width,
                        y=hand_landmarks.landmark[12].y * height,
                        z=hand_landmarks.landmark[12].z
                    ),
                    dip=SimpleNamespace(
                        x=hand_landmarks.landmark[11].x * width,
                        y=hand_landmarks.landmark[11].y * height,
                        z=hand_landmarks.landmark[11].z
                    ),
                    pip=SimpleNamespace(
                        x=hand_landmarks.landmark[10].x * width,
                        y=hand_landmarks.landmark[10].y * height,
                        z=hand_landmarks.landmark[10].z
                    ),
                    mcp=SimpleNamespace(
                        x=hand_landmarks.landmark[9].x * width,
                        y=hand_landmarks.landmark[9].y * height,
                        z=hand_landmarks.landmark[9].z
                    )
                ),

                ring=SimpleNamespace(
                    tip=SimpleNamespace(
                        x=hand_landmarks.landmark[16].x * width,
                        y=hand_landmarks.landmark[16].y * height,
                        z=hand_landmarks.landmark[16].z
                    ),
                    dip=SimpleNamespace(
                        x=hand_landmarks.landmark[15].x * width,
                        y=hand_landmarks.landmark[15].y * height,
                        z=hand_landmarks.landmark[15].z
                    ),
                    pip=SimpleNamespace(
                        x=hand_landmarks.landmark[14].x * width,
                        y=hand_landmarks.landmark[14].y * height,
                        z=hand_landmarks.landmark[14].z
                    ),
                    mcp=SimpleNamespace(
                        x=hand_landmarks.landmark[13].x * width,
                        y=hand_landmarks.landmark[13].y * height,
                        z=hand_landmarks.landmark[13].z
                    )
                ),

                pinky=SimpleNamespace(
                    tip=SimpleNamespace(
                        x=hand_landmarks.landmark[20].x * width,
                        y=hand_landmarks.landmark[20].y * height,
                        z=hand_landmarks.landmark[20].z
                    ),
                    dip=SimpleNamespace(
                        x=hand_landmarks.landmark[19].x * width,
                        y=hand_landmarks.landmark[19].y * height,
                        z=hand_landmarks.landmark[19].z
                    ),
                    pip=SimpleNamespace(
                        x=hand_landmarks.landmark[18].x * width,
                        y=hand_landmarks.landmark[18].y * height,
                        z=hand_landmarks.landmark[18].z
                    ),
                    mcp=SimpleNamespace(
                        x=hand_landmarks.landmark[17].x * width,
                        y=hand_landmarks.landmark[17].y * height,
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

        _landmarks.landmark = SimpleNamespace(
            a=_landmarks.wrist,

            b=_landmarks.fingers.thumb.cmc,
            c=_landmarks.fingers.thumb.mcp,
            d=_landmarks.fingers.thumb.ip,
            e=_landmarks.fingers.thumb.tip,

            f=_landmarks.fingers.index.mcp,
            g=_landmarks.fingers.index.pip,
            h=_landmarks.fingers.index.dip,
            i=_landmarks.fingers.index.tip,

            j=_landmarks.fingers.middle.mcp,
            k=_landmarks.fingers.middle.pip,
            l=_landmarks.fingers.middle.dip,
            m=_landmarks.fingers.middle.tip,

            n=_landmarks.fingers.ring.mcp,
            o=_landmarks.fingers.ring.pip,
            p=_landmarks.fingers.ring.dip,
            q=_landmarks.fingers.ring.tip,

            r=_landmarks.fingers.pinky.mcp,
            s=_landmarks.fingers.pinky.pip,
            t=_landmarks.fingers.pinky.dip,
            u=_landmarks.fingers.pinky.tip
        )

        # Assign an alias for thumb.ip
        _landmarks.fingers.thumb.dip = SimpleNamespace(
            x=_landmarks.fingers.thumb.ip.x,
            y=_landmarks.fingers.thumb.ip.y,
            z=_landmarks.fingers.thumb.ip.z
        )

        _landmarks.mcp_width = SimpleNamespace(
            index_middle=calculate_distance_sn(_landmarks.fingers.index.mcp,
                                               _landmarks.fingers.middle.mcp),
            middle_ring=calculate_distance_sn(_landmarks.fingers.middle.mcp, _landmarks.fingers.ring.mcp),
            ring_pinky=calculate_distance_sn(_landmarks.fingers.ring.mcp, _landmarks.fingers.pinky.mcp)
        )

        _landmarks.fingertip_distance_aggregated = _landmarks.mcp_width.index_middle + \
            _landmarks.mcp_width.middle_ring + _landmarks.mcp_width.ring_pinky

        _landmarks.fingertip_major_axes = SimpleNamespace(
            thumb=calculate_distance_sn(
                _landmarks.fingers.thumb.mcp,
                _landmarks.fingers.thumb.ip
            ) * thumb_width_length_ratio * tip_dip_length_ratio * finger_mcp_width_ratio,
            index=_landmarks.mcp_width.index_middle / 2 * finger_mcp_width_ratio,
            middle=_landmarks.mcp_width.middle_ring / 2 * finger_mcp_width_ratio,
            ring=_landmarks.mcp_width.ring_pinky / 2 * finger_mcp_width_ratio,
            pinky=_landmarks.mcp_width.ring_pinky / 2 * pinky_ring_width_ratio * finger_mcp_width_ratio
        )

        thumb_tip = np.array([_landmarks.fingers.thumb.tip.x, _landmarks.fingers.thumb.tip.y])
        thumb_ip = np.array([_landmarks.fingers.thumb.ip.x, _landmarks.fingers.thumb.ip.y])

        index_tip = np.array([_landmarks.fingers.index.tip.x, _landmarks.fingers.index.tip.y])
        index_dip = np.array([_landmarks.fingers.index.dip.x, _landmarks.fingers.index.dip.y])

        middle_tip = np.array([_landmarks.fingers.middle.tip.x, _landmarks.fingers.middle.tip.y])
        middle_dip = np.array([_landmarks.fingers.middle.dip.x, _landmarks.fingers.middle.dip.y])

        ring_tip = np.array([_landmarks.fingers.ring.tip.x, _landmarks.fingers.ring.tip.y])
        ring_dip = np.array([_landmarks.fingers.ring.dip.x, _landmarks.fingers.ring.dip.y])

        pinky_tip = np.array([_landmarks.fingers.pinky.tip.x, _landmarks.fingers.pinky.tip.y])
        pinky_dip = np.array([_landmarks.fingers.pinky.dip.x, _landmarks.fingers.pinky.dip.y])

        thumb_tip_ip_distance = calculate_distance_array(thumb_tip, thumb_ip)
        index_tip_dip_distance = calculate_distance_array(index_tip, index_dip)
        middle_tip_dip_distance = calculate_distance_array(middle_tip, middle_dip)
        ring_tip_dip_distance = calculate_distance_array(ring_tip, ring_dip)
        pinky_tip_dip_distance = calculate_distance_array(pinky_tip, pinky_dip)

        _landmarks.fingertip_minor_axes = SimpleNamespace(
            thumb=(_landmarks.fingertip_major_axes.thumb + thumb_tip_ip_distance) * 0.5,
            index=(_landmarks.fingertip_major_axes.index + index_tip_dip_distance) * 0.5,
            middle=(_landmarks.fingertip_major_axes.middle + middle_tip_dip_distance) * 0.5,
            ring=(_landmarks.fingertip_major_axes.ring + ring_tip_dip_distance) * 0.5,
            pinky=(_landmarks.fingertip_major_axes.pinky + pinky_tip_dip_distance) * 0.5
        )

        _landmarks.fingertip_angle = SimpleNamespace(
            thumb=np.rad2deg(np.arctan2(thumb_tip[1] - thumb_ip[1], thumb_tip[0] - thumb_ip[0])) + 90,
            index=np.rad2deg(np.arctan2(index_tip[1] - index_dip[1], index_tip[0] - index_dip[0])) + 90,
            middle=np.rad2deg(np.arctan2(middle_tip[1] - middle_dip[1], middle_tip[0] - middle_dip[0])) + 90,
            ring=np.rad2deg(np.arctan2(ring_tip[1] - ring_dip[1], ring_tip[0] - ring_dip[0])) + 90,
            pinky=np.rad2deg(np.arctan2(pinky_tip[1] - pinky_dip[1], pinky_tip[0] - pinky_dip[0])) + 90
        )

        _landmarks_sn.landmarks_list.append(_landmarks)

        if debug_mode.coordination_on:
            _text = f'handedness: {_landmarks.handedness}\n'
            for k, v in _landmarks.fingers.__dict__.items():
                _text += '{} tip: ({:.3f}, {:.3f}, {:.6f})\n'.format(k, v.tip.x, v.tip.y, v.tip.z)

            for i, line in enumerate(_text.split('\n')):
                cv2.putText(
                    img=image,
                    text=line,
                    org=(10, _yy),
                    fontFace=cv2.FONT_HERSHEY_PLAIN,
                    fontScale=1,
                    color=(0x00, 0x00, 0xFF),
                    thickness=1
                )
                _yy += _dy

    _landmarks_sn.image = image


def fetch_orientation(_landmarks_sn):
    for _landmarks in _landmarks_sn.landmarks_list:

        wrist = np.array([_landmarks.wrist.x, _landmarks.wrist.y])
        thumb_cmc = np.array([_landmarks.fingers.thumb.cmc.x, _landmarks.fingers.thumb.cmc.y])
        pinky_mcp = np.array([_landmarks.fingers.pinky.mcp.x, _landmarks.fingers.pinky.mcp.y])

        thumb_cmc_wrist_deg = np.rad2deg(np.arctan2(thumb_cmc[1] - wrist[1], thumb_cmc[0] - wrist[0]))
        pinky_mcp_wrist_deg = np.rad2deg(np.arctan2(pinky_mcp[1] - wrist[1], pinky_mcp[0] - wrist[0]))
        orientation_angle = thumb_cmc_wrist_deg - pinky_mcp_wrist_deg
        if orientation_angle < -180:
            orientation_angle += 360
        elif orientation_angle > 180:
            orientation_angle -= 360

        if _landmarks.handedness == 'Left' and orientation_angle >= 0 \
                or _landmarks.handedness == 'Right' and orientation_angle <= 0:
            _landmarks.orientation = 'Front'
        else:
            _landmarks.orientation = 'Rear'
            _landmarks.finger_status = SimpleNamespace(
                thumb=False,
                index=False,
                middle=False,
                ring=False,
                pinky=False
            )

        if debug_mode.orientation_on:
            coord1 = (int(wrist[0]) + 10, int(wrist[1]))
            _text = 'orientation: {} angle: {}'.format(
                _landmarks.orientation, orientation_angle
            )
            cv2.putText(
                img=_landmarks_sn.image,
                text=_text,
                org=coord1,
                fontFace=cv2.FONT_HERSHEY_PLAIN,
                fontScale=1,
                color=(0x00, 0x00, 0xFF),
                thickness=1
            )

            coord2 = (int(wrist[0]) + 10, int(wrist[1]) + 15)
            _text = 't_w: {} p_w: {}'.format(
                thumb_cmc_wrist_deg, pinky_mcp_wrist_deg
            )
            cv2.putText(
                img=_landmarks_sn.image,
                text=_text,
                org=coord2,
                fontFace=cv2.FONT_HERSHEY_PLAIN,
                fontScale=1,
                color=(0x00, 0x00, 0xFF),
                thickness=1
            )


def fetch_finger_self_occlusion(_landmarks_sn):
    for _landmarks in _landmarks_sn.landmarks_list:
        if _landmarks.orientation == 'Front':
            wrist_tip_distance = {
                'thumb': calculate_distance_sn(_landmarks.wrist, _landmarks.fingers.thumb.tip),
                'index': calculate_distance_sn(_landmarks.wrist, _landmarks.fingers.index.tip),
                'middle': calculate_distance_sn(_landmarks.wrist, _landmarks.fingers.middle.tip),
                'ring': calculate_distance_sn(_landmarks.wrist, _landmarks.fingers.ring.tip),
                'pinky': calculate_distance_sn(_landmarks.wrist, _landmarks.fingers.pinky.tip)
            }
            wrist_cmp_distance = {
                'thumb': calculate_distance_sn(_landmarks.wrist, _landmarks.fingers.thumb.ip),
                'index': calculate_distance_sn(_landmarks.wrist, _landmarks.fingers.index.dip),
                'middle': calculate_distance_sn(_landmarks.wrist, _landmarks.fingers.middle.dip),
                'ring': calculate_distance_sn(_landmarks.wrist, _landmarks.fingers.ring.dip),
                'pinky': calculate_distance_sn(_landmarks.wrist, _landmarks.fingers.pinky.dip)
            }

            for k, v in wrist_tip_distance.items():
                if v < wrist_cmp_distance[k]:
                    _landmarks.finger_status.__dict__[k] = False


def fetch_palm_occlusion(_landmarks_sn):
    # hand occlusion detection
    height, width, _ = _landmarks_sn.image.shape
    ls = _landmarks_sn.landmarks_list
    if len(ls) > 1:
        if ls[0].fingertip_distance_aggregated > ls[1].fingertip_distance_aggregated:
            close, distant = ls[0], ls[1]
        else:
            close, distant = ls[1], ls[0]

        # Landmarks may be out of the frame
        close_x0 = distant_x0 = width * 2
        close_y0 = distant_y0 = height * 2
        close_x1 = distant_x1 = -width
        close_y1 = distant_y1 = -height
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
                img=_landmarks_sn.image,
                text='Close',
                org=(int(close_x0) + 10, int(close_y0) + 10),
                fontFace=cv2.FONT_HERSHEY_PLAIN,
                fontScale=1,
                color=(0x00, 0x00, 0xFF),
                thickness=1
            )
            cv2.rectangle(
                img=_landmarks_sn.image,
                pt1=(int(close_x0), int(close_y0)),
                pt2=(int(close_x1), int(close_y1)),
                color=(0x00, 0xFF, 0x00)
            )

            cv2.putText(
                img=_landmarks_sn.image,
                text='Distant',
                org=(int(distant_x0) + 10, int(distant_y0) + 10),
                fontFace=cv2.FONT_HERSHEY_PLAIN,
                fontScale=1,
                color=(0x00, 0x00, 0xFF),
                thickness=1
            )
            cv2.rectangle(
                img=_landmarks_sn.image,
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
                xi, yi = int(vi.tip.x * 0.7 + vi.dip.x * 0.3), int(vi.tip.y * 0.7 + vi.dip.y * 0.3)
                ri = int(distant.fingertip_major_axes.__dict__[ki] * 2.5)

                for first, second in connections_number:
                    x1 = int(close.landmark.__dict__[landmark_order[first]].x)
                    y1 = int(close.landmark.__dict__[landmark_order[first]].y)
                    x2 = int(close.landmark.__dict__[landmark_order[second]].x)
                    y2 = int(close.landmark.__dict__[landmark_order[second]].y)
                    intersect_res = intersect_line_circle((xi, yi), ri, (x1, y1), (x2, y2))
                    if intersect_res:
                        distant.finger_status.__dict__[ki] = False


def generate_mask(_image_height, _image_width, _center, _angle, _major_axe, _minor_axe):
    y, x = np.ogrid[0: _image_height, 0: _image_width]
    alpha = np.deg2rad(_angle)
    x0, y0 = x - _center[0], y - _center[1]
    sine = math.sin(alpha)
    cosine = math.cos(alpha)
    a, b = _major_axe, _minor_axe
    mask = b * b * ((x0 * cosine + y0 * sine) ** 2) + a * a * ((x0 * sine - y0 * cosine) ** 2) <= a * a * b * b

    return mask


def process_fingertip(_landmarks_sn):
    blur_mode = debug_mode.blur_on
    blur_complete = False
    if _landmarks_sn.landmarks_list:
        _image = _landmarks_sn.image
        image_height, image_width, _ = _image.shape
        mask_image = np.ones(_image.shape, np.int8)
        kernel_size = 35

        if blur_mode == 1:
            blur_source_image = cv2.blur(_image, (kernel_size, kernel_size))

        elif blur_mode == 2:
            blur_source_image = cv2.GaussianBlur(_image, (kernel_size, kernel_size), 0)

        else:
            blur_source_image = None

        for _landmark in _landmarks_sn.landmarks_list:
            for k, v in _landmark.fingers.__dict__.items():
                center = (int(v.tip.x * 0.7 + v.dip.x * 0.3), int(v.tip.y * 0.7 + v.dip.y * 0.3))
                major_axe = int(_landmark.fingertip_major_axes.__dict__[k])
                minor_axe = int(_landmark.fingertip_minor_axes.__dict__[k])
                angle = _landmark.fingertip_angle.__dict__[k]

                if blur_mode == 1 or blur_mode == 2 and _landmark.finger_status.__dict__[k]:
                    mask = generate_mask(
                        _image_height=image_height,
                        _image_width=image_width,
                        _center=center,
                        _angle=angle,
                        _major_axe=major_axe,
                        _minor_axe=minor_axe
                    )
                    mask_image[mask] = [0, 0, 0]
                    blur_complete = True

                if debug_mode.track_on:
                    _text = 'Major: {}, minor: {}, angle: {:.3f}'.format(major_axe, minor_axe, angle)
                    cv2.putText(
                        img=_landmarks_sn.image,
                        text=_text,
                        org=(center[0] + 10, center[1] + 10),
                        fontFace=cv2.FONT_HERSHEY_PLAIN,
                        fontScale=1,
                        color=(0x00, 0x00, 0xFF),
                        thickness=1
                    )

                    if _landmark.finger_status.__dict__[k]:
                        cv2.ellipse(
                            img=_landmarks_sn.image,
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
                            img=_landmarks_sn.image,
                            center=center,
                            axes=(major_axe, minor_axe),
                            angle=angle,
                            startAngle=0,
                            endAngle=360,
                            color=(0x00, 0x00, 0xFF),
                            thickness=2
                        )

        if blur_mode == 1 or blur_mode == 2 and blur_complete:
            mask_image_reverse = np.ones(image.shape, np.int8) - mask_image
            _image = mask_image * _image + mask_image_reverse * blur_source_image
            _landmarks_sn.image = _image


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

        # Rendering results
        if results.multi_hand_landmarks:
            if debug_mode.landmark_on:
                for num, hand in enumerate(results.multi_hand_landmarks):
                    mp_drawing.draw_landmarks(
                        image, hand, mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                        mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
                    )

            landmarks_sn = SimpleNamespace(
                timestamp=int(round(time.time() * 1000)),
            )

            preprocess(landmarks_sn, results, image)

            fetch_orientation(landmarks_sn)

            fetch_finger_self_occlusion(landmarks_sn)

            fetch_palm_occlusion(landmarks_sn)

        else:
            landmarks_sn = SimpleNamespace(
                timestamp=int(round(time.time() * 1000)),
                landmarks_list=[],
                image=image
            )

        process_fingertip(landmarks_sn)

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
                thickness=2
            )

        cv2.imwrite(os.path.join(folder, '{}.jpeg'.format(landmarks_sn.timestamp)), landmarks_sn.image)
        cv2.imshow('Hand Tracking', landmarks_sn.image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
