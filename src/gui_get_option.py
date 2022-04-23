import PySimpleGUI as sg
import cv2
import numpy as np
from PIL import Image
from pathlib import Path


def get_option():
    """
        Get options from user.
    """
    file_path = ""
    # GUI layout
    layout = [
        [
            sg.FileBrowse(key="file_path"),
            sg.Text("File"),
            sg.InputText()
        ],
        [sg.Submit(key="submit"), sg.Cancel("Exit")],
        [sg.Slider(key='blur_value', range=(1, 31), default_value=7, size=(20, 15), orientation='horizontal')],
        [
            sg.Radio(key='normalization', text='归一化平滑', group_id='Blur_option', default=True),
            sg.Radio(key='gaussian', text='高斯平滑', group_id='Blur_option'),
            sg.Radio(key='nope', text='不处理', group_id='Blur_option')
        ]
    ]
    # WINDOW generation
    window = sg.Window("File selection", layout)

    # Event loop
    while True:
        event, values = window.read(timeout=100)
        if event == 'Exit' or event == sg.WIN_CLOSED:
            break
        elif event == 'submit':
            if values[0] == "":
                sg.popup("No file has been entered.")
                event = ""
            else:
                file_path = values[0]
                break
    window.close()
    return values


if __name__ == "__main__":
    option = get_option()
    print(option)
