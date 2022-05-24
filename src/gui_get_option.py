import PySimpleGUI as sg
from cv2 import cv2


def get_option() -> dict:
    """Get options from user.

    The GUI to interact with user to get options.
    Returns:
        dict: a dict containing user's selections
    """
    camera_count = -1
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.grab():
            camera_count += 1
            cap.release()
        else:
            cap.release()
            break

    layout = [
        [
            sg.Text(text='摄像头输入', font='Any 15'),
            sg.Slider(key='camera_input_no', range=(-1, camera_count), default_value=-1, size=(85, 15),
                      orientation='horizontal')
        ],
        [
            sg.Text(text='文件输入', font='Any 15'),
            sg.InputText(key='text_file_path', size=(90, 15), font='Any 15'),
            sg.FileBrowse(key='file_path', button_text='浏览', font='Any 15'),
        ],
        [
            sg.Text('模糊程度', font='Any 15'),
            sg.Slider(key='blur_value', range=(1, 51), default_value=11, size=(90, 15), orientation='horizontal')
        ],
        [
            sg.Text('模糊处理', font='Any 15'),
            sg.Radio(key='random', text='随机模糊', group_id='blur_option', default=True, font='Any 15'),
            sg.Radio(key='averaging', text='平均平滑', group_id='blur_option', font='Any 15'),
            sg.Radio(key='gaussian', text='高斯模糊', group_id='blur_option', font='Any 15'),
            sg.Radio(key='median', text='中值滤波', group_id='blur_option', font='Any 15'),
            sg.Radio(key='bilateral', text='双边滤波', group_id='blur_option', font='Any 15'),
            sg.Radio(key='nope', text='不处理', group_id='blur_option', font='Any 15')
        ],
        [
            sg.Text('编码格式', font='Any 15'),
            sg.Radio(key='h265', text='H.265', group_id='codec', default=True, font='Any 15'),
            sg.Radio(key='h264', text='H.264', group_id='codec', font='Any 15')
        ],
        [
            sg.Text('输出速度', font='Any 15'),
            sg.Radio(key='ultrafast', text='ultrafast', group_id='preset', font='Any 15'),
            sg.Radio(key='superfast', text='superfast', group_id='preset', font='Any 15'),
            sg.Radio(key='faster', text='faster', group_id='preset', font='Any 15'),
            sg.Radio(key='fast', text='fast', group_id='preset', default=True, font='Any 15'),
            sg.Radio(key='medium', text='medium', group_id='preset', font='Any 15'),
            sg.Radio(key='slow', text='slow', group_id='preset', font='Any 15'),
            sg.Radio(key='slower', text='slower', group_id='preset', font='Any 15'),
            sg.Radio(key='veryslow', text='veryslow', group_id='preset', font='Any 15'),
        ],
        [
            sg.Text('处理加速', font='Any 15'),
            sg.Radio(key='multi_process', text='多进程', group_id='multiple', default=True, font='Any 15'),
            sg.Radio(key='multi_thread', text='多线程', group_id='multiple', font='Any 15'),
            sg.Radio(key='single_process', text='单进程', group_id='multiple', font='Any 15'),
            sg.Radio(key='single_thread', text='单线程', group_id='multiple', font='Any 15')
        ],
        [
            sg.Submit(key='submit', button_text='提交', font='Any 15'),
            sg.Cancel(key='exit', button_text='退出', font='Any 15')
        ]
    ]
    # WINDOW generation
    window = sg.Window('请选择输入流', layout)

    # Event loop
    while True:
        event, values = window.read(timeout=100)
        if event == 'exit' or event == sg.WIN_CLOSED:
            break
        elif event == 'submit':
            if values['text_file_path'] == '' and values['camera_input_no'] < 0:
                sg.popup('未选择输入流', font='Any 15')
                event = ''
            else:
                break
    window.close()
    return values


if __name__ == '__main__':
    option = get_option()
    print(option)
