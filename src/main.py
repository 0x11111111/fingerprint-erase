import mediapipe
import os
import cv2
import time
import multiprocessing as mp
from functools import partial
from types import SimpleNamespace

from gui_get_option import get_option

debug_mode = SimpleNamespace(
    circle_on=False,
    landmark_on=False,
    coordination_on=False,
    output_on=False,
    orientation_on=False,
    frame_rate_on=True,
    scoop_on=False,
)

args = SimpleNamespace(
    folder=os.path.join('../.tmp', '{}'.format(int(round(time.time() * 1000))))
)

mp_drawing = mediapipe.solutions.drawing_utils
mp_hands = mediapipe.solutions.hands

if not os.path.exists("../.tmp"):
    os.mkdir("../.tmp")

os.mkdir(args.folder)

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

landmark_order = 'abcdefghijklmnopqrstu'
image_list = []
num_processes = 4
args.file_path = '../test/4.mp4'

cap = cv2.VideoCapture(args.file_path)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()
num_processes = mp.cpu_count()
args.frame_jump_unit = frame_count // num_processes

if __name__ == '__main__':
    from core_neomask_finger_tracking import fingerprint_erase

    # selection = get_option()
    # args.file_path = selection['file_path']
    # # Kernel size of Gaussian should be odd only
    # args.kernel_size = int(selection['blur_value'] // 2 * 2 + 1)

    # args.blur_mode = 0
    # if selection['normalization']:
    #     args.blur_mode = 1
    # elif selection['gaussian']:
    #     args.blur_mode = 2

    # cap = cv2.VideoCapture(selection['file_path'])

    # frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # cap.release()
    # num_processes = mp.cpu_count()
    # args.frame_jump_unit = frame_count // num_processes

    p = mp.Pool(num_processes)
    p.map(partial(fingerprint_erase, args), range(num_processes))
