# https://yanwei-liu.medium.com/pyautogui-%E4%BD%BF%E7%94%A8python%E6%93%8D%E6%8E%A7%E9%9B%BB%E8%85%A6-662cc3b18b80
# https://pyautogui.readthedocs.io/en/latest/keyboard.html

import pyautogui as pag
import keyboard
import mouseinfo
from cv2 import cv2 as cv2
import numpy as np
from PIL import Image
import time
import threading
from cv2 import cv2 as cv2
import pytesseract as pyocr

escFlag = 0
skill_flag = [0] * 6
shift_flag = 0
speed = 0
center = [300, 300] 
state = 0

map_dict = {"汪四": "1"}

def Exit():
    global escFlag
    while escFlag == 0:  # making a loop
        try:  
            if keyboard.is_pressed('ESC'):  
                escFlag = 1
                break  
            else:
                time.sleep(0.5)
                pass
        except:
            break  

def FlagController():
    global skill_flag, shift_flag
    while escFlag == 0:
        time.sleep(1)
        for i in range(len(skill_flag)):
            if skill_flag[i] >= 1:
                skill_flag[i] -= 1
        if shift_flag >= 1:
            shift_flag -= 1
        # print(skill_flag)

def ShiftController():
    global shift_flag, speed
    while escFlag == 0:
        if speed == 1:
            if shift_flag == 0:
                pag.press('shiftleft')
                shift_flag = 2
            speed = 0

def show(im, islist = False):
	if islist:
		for i in im:
			testimage = Image.fromarray(i)
			testimage.show()
	else :
		testimage = Image.fromarray(im)
		testimage.show()

def MouthThread(angle):
    pag.moveRel(angle*2, 0, abs(float(angle/180)))
    # print('Mouse %f' %angle)
    # time.sleep(1)

def KeyboardThread(dist):
    global speed
    time = float(dist/100)
    if dist > 100:
        speed = 1
        time = (time - 0.5) / 1.5 
    pag.press('w', interval=time) 
    print('Move toward %f' %dist)
    # time.sleep(1)


def FindRed(img) :
    lower = np.array([0, 0, 200])
    upper = np.array([60, 60, 255])

    filtered = cv2.inRange(img, lower, upper)
    blurred = cv2.GaussianBlur(filtered, (5, 5), 0)
    cnts, _ = cv2.findContours(blurred.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rtimg = np.zeros_like(img)
    point_list = []
    for cnt in cnts:
        # compute the (rotated) bounding box around then
        # contour and then draw it		
        rect = np.int32(cv2.boxPoints(cv2.minAreaRect(cnt)))
        rtimg = cv2.drawContours(img, [rect], -1, (0, 255, 0), 2)
        point = [rect[2][0] + (rect[0][0] - rect[1][0])/2, rect[2][1] + (rect[1][1]- rect[2][1])/2]
        point_list.append(point)
    return rtimg, point_list

    
def CatchScreen(x, y, w, h):
    tStart = time.time()
    im = pag.screenshot('screen.png',region=(x, y, w, h))
    img = cv2.imread('screen.png')
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # threshold = cv2.adaptiveThreshold(img, 255, cv2.THRESH_BINARY, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 11, 7)
    # _ , threshold = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    img = cv2.resize(img, (300, 92))
    text = pyocr.image_to_string(img, lang='chi_tra', config='--psm 7 --oem 3')
    tEnd = time.time()
    print('Total OCR Time = ' + str(float(tEnd - tStart)))
    print(text.replace(' ', ''))
    # print('ScreenShot Time = ' + str(float(tEnd - tStart)))
    # im.show()
    # show(img)
    return text.replace(' ', '')
        
def CatchMap(x, y, w, h):
    pag.screenshot('map.png',region=(x, y, w, h))
    img = cv2.imread('map.png')
    return img

def GetScreenPos():
    while True:
        time.sleep(0.5)
        x,y = pag.position()
        print('%f, %f' %(x,y))

def Skill (i = -1, key = '', time = 0):
    global skill_flag
    if i == -1:
        if skill_flag[0] == 0:
            pag.press('f')
            skill_flag[0] = 10 
        elif skill_flag[1] == 0:
            pag.press('!')
            skill_flag[1] = 6 
        elif skill_flag[2] == 0:
            pag.press('n')
            skill_flag[2] = 16 
        elif skill_flag[3] == 0:
            pag.press('r')
            skill_flag[3] = 26 
        # elif skill_flag[4] == 0:
        #     pag.press('t')
        #     skill_flag[4] = 500 
        else :
            pass
    else :
        if skill_flag[i] == 0:
            pag.press(key)
            skill_flag[i] = time


def MoveTo(point1, point2, toPoint = True) :  # Center to Point
    vec1 = np.array([0.0,-10.0])
    dx = point2[0] - point1[0]
    dy = point2[1] - point1[1]
    vec2 = np.array([dx,dy])
    Lv1=np.sqrt(vec1.dot(vec1))
    Lv2=np.sqrt(vec2.dot(vec2))     # dist
    cos_angle=vec1.dot(vec2)/(Lv1*Lv2)
    angle=np.arccos(cos_angle)
    angle2=angle*360/2/np.pi
    if vec2[0] < 0:
        angle2 *= -1                # angle
    # Move
    if toPoint == False:
        Lv2 -= 20

    mouth_thread = threading.Thread(target = MouthThread, args = (angle2, ))
    keyboard_thread = threading.Thread(target = KeyboardThread, args = (Lv2, ))
    # print(vec1)
    # print(vec2)
    mouth_thread.start()
    keyboard_thread.start()
    if Lv2 >= 100:
        Skill(5, 'shift', 2)
    mouth_thread.join()
    keyboard_thread.join()

def Map() :
    # text = CatchScreen(880, 33 , 85, 23)
    text = '火山口M吐'
    ret = -1
    if '汪四' in text or '野' in text:
        print('Map:颶風荒野')
        ret = 1
    elif '同' in text or '祭壇' in text:
        print('Map:風化祭壇')
        ret = 2
    elif '深海' in text or '瀑布' in text or '了放' in text or '計生' in text:
        print('Map:深海瀑布')
        ret = 3
    elif '火山' in text or '口' in text or '國' in text:
        print('Map:火山口地面')
        ret = 4
    elif 'pm' in text or 'um' in text or '開' in text:
        print('Map:銀光藤蔓')
        ret = 5
    elif '俘' in text or '審' in text:
        print('Map:俘虜的審判')
        ret = 6
    elif '由記' in text or '由六' in text or '由訂' in text:
        print('Map:沙沼')
        ret = 7
    elif '下和' in text or '下生' in text:
        print('Map:遇難船隻')
        ret = 8
    elif '陽' in text or '景' in text or '台' in text:
        print('Map:夕陽觀景台')
        ret = 0
    else :
        print('no recg')
    return ret

def Atttack(enemy) :
    global center
    MoveTo(center, enemy, toPoint=False)
    Skill()

def State0 () :
    global center, state
    door = [center[0], center[1] + 150]
    MoveTo(center, door)                # 貼門
    pag.click(x=500, y=500, duration=2) # 點擊進入
    time.sleep(5)
    state += 1

def State1 () :
    global center, state
    mapNo = Map()
    if mapNo == 1:
        point = [10, -200]
        MoveTo(center, point)
    elif mapNo == 1:
        point = [10, -200]
        MoveTo(center, point)
    elif mapNo == 1:
        point = [10, -200]
        MoveTo(center, point)
    elif mapNo == 1:
        point = [10, -200]
        MoveTo(center, point)
    elif mapNo == 1:
        point = [10, -200]
        MoveTo(center, point)
    elif mapNo == 1:
        point = [10, -200]
        MoveTo(center, point)
    elif mapNo == 1:
        point = [10, -200]
        MoveTo(center, point)
    elif mapNo == 1:
        point = [10, -200]
        MoveTo(center, point)
    else:
        print('No Map, Stop at state 1')
    
    Skill(0, 'f', 10)
    point = [0, 30]
    MoveTo(center, point)
    pag.press('z', presses=3, interval=0.5)
    state += 1


def State2 () :
    global center, state
    text = CatchScreen(250, 500, 150,50) # F12的框框
    if len(text) != 0:      # 有F12
        pag.press('f12')
        time.sleep(0.3)
        pag.click(100,100,duration=0.8)
        time.sleep(1.0)
        map = Map()
        if map == -1 or map == 0: # 成功返回入口
            state = 0
        else :
            state = 2
    else :                  # 沒有F12
        state += 1

def State3  () :
    global state
    img = CatchMap(830, 60, 130, 130)        # 填入地圖位置
    enemy = FindRed(img)
    if len(enemy) > 0 :
        Atttack(enemy[-1])
    else :
        state = 2
        time.sleep(1)


def AutoMove() :
    global state    # 0:觀景台 點開始 1:抓地圖 進場 打 撿 2:等一下 看F12 3:活動(有沒有紅點 Y:打->打完回到state0 N:根據地圖走->回到state1)
    if state == 0 :
        State0()
    elif state == 1:
        State1()
    elif state == 2:
        State2()
    elif state == 3:
        State3()
    else:
        time.sleep(1)
        pass
    
    # center = [303, 294]             # 要修改
    # red = cv2.imread('red.png')     # 要修改
    # _, point_list = FindRed(red)
    # # text = CatchScreen(880, 33 , 85, 23)
    # # time.sleep(1)
    
    # for point in point_list:
    #     MoveTo(center, point, toPoint = False)
    #     Skill(0, 'f', 10)
    # time.sleep(1)


if __name__ == "__main__":
    # GetScreenPos() # 818 33 1026 63
    # text = CatchScreen(842, 38 , 130, 16)
    # Map()

    print('Press ESC to exit')
    flagThread = threading.Thread(target = FlagController, args = ())
    flagThread.start()

    escThread = threading.Thread(target = Exit, args = ())
    escThread.start()

    speedThread = threading.Thread(target = ShiftController, args = ())
    speedThread.start()

    # AutoMove()
    moveFlag = 0
    while escFlag == 0:  # making a loop
        try:  
            if keyboard.is_pressed('/'):  
                moveFlag = 1
                print('Start')
                break  
            else:
                pass
        except:
            break  
    
    while escFlag == 0:  # making a loop
        try:  
            if moveFlag == 1:
                AutoMove()
            else:
                pass
        except:
            break  
    

    flagThread.join()
    escThread.join()
    speedThread.join()
