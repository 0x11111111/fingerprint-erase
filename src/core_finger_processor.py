import cv2
import mediapipe
import numpy as np

from types import SimpleNamespace
from utility import calculate_distance_sn, calculate_distance_array, intersect_line_circle


def preprocess(_landmarks_sn, res, _image, info):
    height, width, _ = _image.shape
    _landmarks_sn.landmarks_list = []
    _yy, _dy = 10, 15
    for idx, hand_landmarks in enumerate(res.multi_hand_landmarks):
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

        _landmarks.mcp_width_sum = _landmarks.mcp_width.index_middle + \
            _landmarks.mcp_width.middle_ring + _landmarks.mcp_width.ring_pinky

        _landmarks.fingertip_minor_axe = SimpleNamespace(
            thumb=calculate_distance_sn(
                _landmarks.fingers.thumb.mcp,
                _landmarks.fingers.thumb.ip
            ) * info.thumb_width_length_ratio * info.tip_dip_length_ratio * info.finger_mcp_width_ratio,
            index=_landmarks.mcp_width.index_middle / 2 * info.finger_mcp_width_ratio,
            middle=_landmarks.mcp_width.middle_ring / 2 * info.finger_mcp_width_ratio,
            ring=_landmarks.mcp_width.ring_pinky / 2 * info.finger_mcp_width_ratio,
            pinky=_landmarks.mcp_width.ring_pinky / 2 * info.pinky_ring_width_ratio * info.finger_mcp_width_ratio
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

        _landmarks.fingertip_major_axe = SimpleNamespace(
            thumb=(_landmarks.fingertip_minor_axe.thumb + thumb_tip_ip_distance) * 0.5,
            index=(_landmarks.fingertip_minor_axe.index + index_tip_dip_distance) * 0.5,
            middle=(_landmarks.fingertip_minor_axe.middle + middle_tip_dip_distance) * 0.5,
            ring=(_landmarks.fingertip_minor_axe.ring + ring_tip_dip_distance) * 0.5,
            pinky=(_landmarks.fingertip_minor_axe.pinky + pinky_tip_dip_distance) * 0.5
        )

        _landmarks.fingertip_angle = SimpleNamespace(
            thumb=np.rad2deg(np.arctan2(thumb_tip[1] - thumb_ip[1], thumb_tip[0] - thumb_ip[0])),
            index=np.rad2deg(np.arctan2(index_tip[1] - index_dip[1], index_tip[0] - index_dip[0])),
            middle=np.rad2deg(np.arctan2(middle_tip[1] - middle_dip[1], middle_tip[0] - middle_dip[0])),
            ring=np.rad2deg(np.arctan2(ring_tip[1] - ring_dip[1], ring_tip[0] - ring_dip[0])),
            pinky=np.rad2deg(np.arctan2(pinky_tip[1] - pinky_dip[1], pinky_tip[0] - pinky_dip[0]))
        )

        _landmarks_sn.landmarks_list.append(_landmarks)

        if info.flags.coordination_on:
            _text = f'handedness: {_landmarks.handedness}\n'
            for k, v in _landmarks.fingers.__dict__.items():
                _text += '{} tip: ({:.3f}, {:.3f}, {:.6f})\n'.format(k, v.tip.x, v.tip.y, v.tip.z)

            for i, line in enumerate(_text.split('\n')):
                cv2.putText(
                    img=_image,
                    text=line,
                    org=(10, _yy),
                    fontFace=cv2.FONT_HERSHEY_PLAIN,
                    fontScale=1,
                    color=(0x00, 0x00, 0xFF),
                    thickness=2
                )
                _yy += _dy

    _landmarks_sn.image = _image


def detect_orientation(_landmarks_sn, info):
    for _landmarks in _landmarks_sn.landmarks_list:

        wrist = np.array([_landmarks.wrist.x, _landmarks.wrist.y])
        thumb_cmc = np.array([_landmarks.fingers.thumb.cmc.x, _landmarks.fingers.thumb.cmc.y])
        pinky_mcp = np.array([_landmarks.fingers.pinky.mcp.x, _landmarks.fingers.pinky.mcp.y])

        thumb_cmc_wrist_deg = np.rad2deg(np.arctan2(thumb_cmc[1] - wrist[1], thumb_cmc[0] - wrist[0]))
        pinky_mcp_wrist_deg = np.rad2deg(np.arctan2(pinky_mcp[1] - wrist[1], pinky_mcp[0] - wrist[0]))
        orientation_angle = thumb_cmc_wrist_deg - pinky_mcp_wrist_deg
        t_p_angle = orientation_angle
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

        if info.flags.orientation_on:
            coord1 = (int(wrist[0]) + 10, int(wrist[1]))
            _text = 'orientation: {} t_p_angle: {:.5f}'.format(
                _landmarks.orientation, t_p_angle
            )
            cv2.putText(
                img=_landmarks_sn.image,
                text=_text,
                org=coord1,
                fontFace=cv2.FONT_HERSHEY_PLAIN,
                fontScale=1,
                color=(0x00, 0x00, 0xFF),
                thickness=2
            )

            coord2 = (int(wrist[0]) + 10, int(wrist[1]) + 40)
            _text = 't_w: {:.3f} p_w: {:.3f}'.format(
                thumb_cmc_wrist_deg, pinky_mcp_wrist_deg
            )
            cv2.putText(
                img=_landmarks_sn.image,
                text=_text,
            org=coord2,
                fontFace=cv2.FONT_HERSHEY_PLAIN,
                fontScale=1,
                color=(0x00, 0x00, 0xFF),
                thickness=2
            )


def detect_finger_self_occlusion(_landmarks_sn, info):
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


def detect_palm_occlusion(_landmarks_sn, info):
    # hand occlusion detection
    height, width, _ = _landmarks_sn.image.shape
    ls = _landmarks_sn.landmarks_list
    mp_hands = mediapipe.solutions.hands

    if len(ls) > 1:
        if ls[0].mcp_width_sum > ls[1].mcp_width_sum:
            close, distant = ls[0], ls[1]
        else:
            close, distant = ls[1], ls[0]

        # Landmarks may be out of the frame
        close_x0 = distant_x0 = width * 2
        close_y0 = distant_y0 = height * 2
        close_x1 = distant_x1 = -width
        close_y1 = distant_y1 = -height
        for i in info.landmark_order:
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

        if info.flags.box_on:
            cv2.putText(
                img=_landmarks_sn.image,
                text='Close',
                org=(int(close_x0) + 10, int(close_y0) + 10),
                fontFace=cv2.FONT_HERSHEY_PLAIN,
                fontScale=2,
                color=(0x00, 0x00, 0xFF),
                thickness=2
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
                fontScale=2,
                color=(0x00, 0x00, 0xFF),
                thickness=2
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
                ri = int(distant.fingertip_minor_axe.__dict__[ki] * 2.5)

                for first, second in connections_number:
                    x1 = int(close.landmark.__dict__[info.landmark_order[first]].x)
                    y1 = int(close.landmark.__dict__[info.landmark_order[first]].y)
                    x2 = int(close.landmark.__dict__[info.landmark_order[second]].x)
                    y2 = int(close.landmark.__dict__[info.landmark_order[second]].y)
                    intersect_res = None
                    if not (-info.EPS <= x1 - x2 <= info.EPS and -info.EPS <= y1 - y2 <= info.EPS):
                        intersect_res = intersect_line_circle((xi, yi), ri, (x1, y1), (x2, y2))
                    else:
                        # Two points overlapping as one cannot discriminate one line.
                        return (xi - x1) ** 2 + (yi - y1) ** 2 <= ri ** 2

                    if intersect_res:
                        distant.finger_status.__dict__[ki] = False


def process_fingertip(_landmarks_sn, _blur_mode, _kernel_size, info):
    _image = _landmarks_sn.image
    image_height, image_width, _ = _image.shape
    mask_image = np.zeros(_image.shape, _image.dtype)

    if _blur_mode == 'averaging':
        blur_source_image = cv2.blur(_image, (_kernel_size, _kernel_size))
    elif _blur_mode == 'gaussian':
        blur_source_image = cv2.GaussianBlur(_image, (_kernel_size, _kernel_size), 0)
    elif _blur_mode == 'median':
        blur_source_image = cv2.medianBlur(_image, _kernel_size)
    elif _blur_mode == 'bilateral':
        blur_source_image = cv2.bilateralFilter(_image, _kernel_size, _kernel_size * 2, _kernel_size * 2)
    elif _blur_mode == 'nope':
        blur_source_image = None

    for _landmark in _landmarks_sn.landmarks_list:
        for k, v in _landmark.fingers.__dict__.items():
            center = (int(v.tip.x * 0.7 + v.dip.x * 0.3), int(v.tip.y * 0.7 + v.dip.y * 0.3))
            minor_axe = int(_landmark.fingertip_minor_axe.__dict__[k])
            major_axe = int(_landmark.fingertip_major_axe.__dict__[k])
            angle = _landmark.fingertip_angle.__dict__[k]

            if info.flags.output_on:
                _text = 'Major: {}, minor: {}, angle: {:.3f}'.format(major_axe, minor_axe, angle)
                cv2.putText(
                    img=_image,
                    text=_text,
                    org=(center[0] + 10, center[1] + 10),
                    fontFace=cv2.FONT_HERSHEY_PLAIN,
                    fontScale=1,
                    color=(0x00, 0x00, 0xFF),
                    thickness=2
                )

            if _landmark.finger_status.__dict__[k]:
                if info.flags.circle_on:
                    cv2.ellipse(
                        img=_image,
                        center=center,
                        axes=(major_axe, minor_axe),
                        angle=angle,
                        startAngle=0,
                        endAngle=360,
                        color=(0x00, 0xFF, 0x00),
                        thickness=2
                    )

                if blur_source_image is not None:
                    cv2.ellipse(
                        img=mask_image,
                        center=center,
                        axes=(major_axe, minor_axe),
                        angle=angle,
                        startAngle=0,
                        endAngle=360,
                        color=(0xFF, 0xFF, 0xFF),
                        thickness=-1
                    )

            else:
                if info.flags.circle_on:
                    cv2.ellipse(
                        img=_image,
                        center=center,
                        axes=(major_axe, minor_axe),
                        angle=angle,
                        startAngle=0,
                        endAngle=360,
                        color=(0x00, 0x00, 0xFF),
                        thickness=2
                    )

    if _blur_mode in ('averaging', 'gaussian', 'median', 'bilateral'):
        _landmarks_sn.image = np.where(mask_image > 0, blur_source_image, _image)
    else:
        _landmarks_sn.image = _image
