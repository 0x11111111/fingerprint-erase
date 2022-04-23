import PySimpleGUI as sg
import cv2
import numpy as np
from PIL import Image
from pathlib import Path


def file_read():
    """
        Select a file to read
    """
    file_path = ""
    # GUI layout
    layout = [
        [
            sg.FileBrowse(key="file"),
            sg.Text("File"),
            sg.InputText()
        ],
        [sg.Submit(key="submit"), sg.Cancel("Exit")]
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
    return file_path
