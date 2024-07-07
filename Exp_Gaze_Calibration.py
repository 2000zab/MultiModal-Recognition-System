from psychopy import visual, event, core
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from Configs import GazeReceiver_Configs as config
import socket
import json
import time

def GazeCalibrate(win):
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect((
        config.network_config.TRIGGER_IP,
        config.network_config.TRIGGER_PORT
    ))
    client.send(
        json.dumps(['sess_start', {'timestamp': time.time()}]).encode()
    )
    def quit_cali():
        client.send(
            json.dumps(['sess_end', {'timestamp': time.time()}]).encode()
        )
        win.close()
        core.quit()
    event.globalKeys.add(key='escape', func=quit_cali)
    # 窗口大小
    width, height = win.size
    # 鼠标监视器
    mouse = event.Mouse(visible=True, newPos=(0,0), win=win)
    # 定义校准点
    calibration_points = [
        [width//2, height//2],      # 中心点
        [width//5, height//8],      # 左上角
        [width//5, height//8*7],    # 左下角
        [width//5*4, height//8],    # 右上角
        [width//5*4, height//8*7],  # 右下角
    ]
    # 由于校准点和psychopy坐标不一致，需要调整
    psycho_points = [[x - width // 2, height // 2 - y]for x, y in calibration_points]

    # 定义开始校准
    button = visual.Rect(win, width=500, height=100, pos=(0, 0), fillColor='#2358c3')
    button_text = visual.TextStim(win, text='开始校准', pos=(0, 0), height=60, color='white')
    button.draw()
    button_text.draw()
    win.flip()
    mouse.clickReset()
    while True:
        if button.contains(mouse.getPos()):
            button.fillColor = 'green'
            button.draw()
            button_text.draw()
            win.flip()
            if mouse.getPressed()[0]:
                break
        else:
            button.fillColor = '#2358c3'
            button.draw()
            button_text.draw()
            win.flip()
        if 'escape' in event.getKeys():
            win.close()
            core.quit()
    win.flip()
    core.wait(0.5)

    # 针对每个校准点进行校准
    for i in range(len(psycho_points)):
        true_point = calibration_points[i]
        psycho_point = psycho_points[i]
        # 定义校准点
        point = visual.Circle(win, radius=7, pos=psycho_point, fillColor='red', lineColor='white')
        point.draw()
        win.flip()
        # 等待点击
        while True:
            if point.contains(mouse.getPos()):
                if mouse.getPressed()[0]:
                    # 校准
                    client.send(
                        json.dumps(['cali', {'timestamp': time.time(), 'data': true_point}]).encode()
                    )
                    break
            if 'escape' in event.getKeys():
                win.close()
                core.quit()
        win.flip()
        core.wait(0.5)
        
    # 定义结束校准
    button = visual.Rect(win, width=500, height=100, pos=(0, 0), fillColor='#2358c3')
    button_text = visual.TextStim(win, text='结束校准', pos=(0, 0), height=60, color='white')
    button.draw()
    button_text.draw()
    win.flip()
    mouse.clickReset()
    while True:
        if button.contains(mouse.getPos()):
            button.fillColor = 'green'
            button.draw()
            button_text.draw()
            win.flip()
            if mouse.getPressed()[0]:
                # 结束校准
                client.send(
                    json.dumps(['sess_end', {'timestamp': time.time()}]).encode()
                )
                break
        else:
            button.fillColor = '#2358c3'
            button.draw()
            button_text.draw()
            win.flip()
        if 'escape' in event.getKeys():
            win.close()
            core.quit()
    win.flip()
    core.wait(0.5)
    
if __name__ == '__main__':
    from psychopy import visual, event, core
    win = visual.Window([1920, 1080], fullscr=True, color='black')
    GazeCalibrate(win)
    win.close()
    core.quit()