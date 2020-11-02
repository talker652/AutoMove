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

timeFlag = 0
escFlag = 0
skill_flag = [0] * 4

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
    global timeFlag
    global skill_flag
    while escFlag == 0:
        time.sleep(1)
        for i in range(len(skill_flag)):
            skill_flag[i] -= 1
            if skill_flag[i] < 0:
                skill_flag[i] = 0
        print(skill_flag)


def show(im, islist = False):
	if islist:
		for i in im:
			testimage = Image.fromarray(i)
			testimage.show()
	else :
		testimage = Image.fromarray(im)
		testimage.show()

def AngleAndDistance(point1, point2) :
    vec1 = np.array([0,-10])
    dx = point2[0] - point1[0]
    dy = point2[1] - point1[1]
    vec2 = np.array([dx,dy])
    Lv1=np.sqrt(vec1.dot(vec1))
    Lv2=np.sqrt(vec2.dot(vec2))
    cos_angle=vec1.dot(vec2)/(Lv1*Lv2)
    angle=np.arccos(cos_angle)
    angle2=angle*360/2/np.pi - 180
    if point2[0] - point1[0] < 0:
        angle2 *= -1
    # print(angle2)
    return angle2, Lv2


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

def GetScreenPos():
    global timeFlag
    while timeFlag <= 2:
        time.sleep(0.5)
        x,y = pag.position()
        print('%f, %f' %(x,y))


def MouseMove(angle) :
    pag.moveRel(angle*2, 0, abs(float(angle/180)))

def KeyboardPress(dist):
    time = float(dist/100)
    print('Press Time = %f' %time)
    pag.press('w', interval=time)

def Skill (i = -1, key = None, time = 0):
    global skill_flag
    if i == -1:
        if skill_flag[0] == 0:
            pag.press('t')
            skill_flag[0] = 500 
        elif skill_flag[0] == 0:
            pag.press('f')
            skill_flag[0] = 10 
        elif skill_flag[0] == 0:
            pag.press('!')
            skill_flag[0] = 5 
        elif skill_flag[0] == 0:
            pag.press('n')
            skill_flag[0] = 15 
        elif skill_flag[0] == 0:
            pag.press('r')
            skill_flag[0] = 26 
        else :
            pass
    else :
        pag.press(key)
        skill_flag[i] = time

def Act(angle, dist):
    MouseMove(angle)
    KeyboardPress(dist)
    print('Move To dir:%.0f dist:%.0f' %(angle, dist))

def AutoMove() :
    # center = [303, 294]
    # red = cv2.imread('red.png')
    # img, point_list = FindRed(red)
    text = CatchScreen(880, 33 , 85, 23)
    time.sleep(1)
    
    # for point in point_list:
    #     angle, dist = AngleAndDistance(center, point)
    #     Act(angle, dist)
    # time.sleep(1)


if __name__ == "__main__":
    # GetScreenPos() # 818 33 1026 63
    # text = CatchScreen(842, 38 , 130, 16)

    print('Press ESC to exit')
    flagThread = threading.Thread(target = FlagController, args = ())
    flagThread.start()

    escThread = threading.Thread(target = Exit, args = ())
    escThread.start()

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
