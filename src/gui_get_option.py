import PySimpleGUI as sg


def get_option():
    '''
        Get options from user.
    '''
    file_path = ''
    # GUI layout
    layout = [
        [
            sg.Text('文件选择'),
            sg.InputText(key='text_file_path', size=(90, 15)),
            sg.FileBrowse(key='file_path', button_text='浏览'),
            sg.Submit(key='submit', button_text='提交'),
            sg.Cancel(key='exit', button_text='退出')
        ],
        [sg.Text('模糊程度'), sg.Slider(key='blur_value', range=(1, 51), default_value=7, size=(90, 15), orientation='horizontal')],
        [
            sg.Text('模糊处理'),
            sg.Radio(key='normalization', text='归一化平滑', group_id='Blur_option', default=True),
            sg.Radio(key='gaussian', text='高斯平滑', group_id='Blur_option'),
            sg.Radio(key='nope', text='不处理', group_id='Blur_option')
        ],
        [
            sg.Text('输出画质'),
            sg.Radio(key='mpeg4', text='mpeg4(.mp4) 低画质', group_id='codec', default=True),
            sg.Radio(key='libx264', text='libx264(.mp4) 中画质', group_id='codec'),
            sg.Radio(key='rawvideo', text='rawvideo(.avi) 高画质', group_id='codec'),
            sg.Radio(key='png', text='png(.avi) 高画质', group_id='codec'),
            sg.Radio(key='libvpx', text='libvpx(.webm) 网页', group_id='codec'),
            sg.Radio(key='libvorbis', text='libvorbis(.ogv) 网页', group_id='codec')
        ],
        [
            sg.Text('处理加速'),
            sg.Radio(key='multi_process', text='多进程(长视频)', group_id='multiple', default=True),
            sg.Radio(key='multi_thread', text='多线程(短视频)', group_id='multiple'),
            sg.Radio(key='single_process', text='单进程(备选1)', group_id='multiple'),
            sg.Radio(key='single_thread', text='单线程(备选2)', group_id='multiple'),
        ]
    ]
    # WINDOW generation
    window = sg.Window('请选择视频文件', layout)

    # Event loop
    while True:
        event, values = window.read(timeout=100)
        if event == 'exit' or event == sg.WIN_CLOSED:
            break
        elif event == 'submit':
            if values['text_file_path'] == '':
                sg.popup('未选择文件')
                event = ''
            else:
                file_path = values['text_file_path']
                break
    window.close()
    return values


if __name__ == '__main__':
    option = get_option()
    print(option)
