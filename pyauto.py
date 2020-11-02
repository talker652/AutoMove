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

def FlagController():
    global timeFlag
    while timeFlag <= 2:
        time.sleep(1)
        timeFlag += 1
        print(timeFlag)


def show(im, islist = False):
	if islist:
		for i in im:
			testimage = Image.fromarray(i)
			testimage.show()
	else :
		testimage = Image.fromarray(im)
		testimage.show()

def MouseMove(angle) :
    pag.moveRel(angle*2, 0, abs(float(angle/180)))


def Angle(point1, point2) :
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
    return angle2


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

def AutoMove() :
    center = [303, 294]
    red = cv2.imread('red.png')
    img, point_list = FindRed(red)
    for point in point_list:
        angle = Angle(center, point)
        print(angle)
        MouseMove(angle)
    
def CatchScreen(x, y, w, h):
    tStart = time.time()
    im = pag.screenshot('screen.png',region=(x, y, w, h))
    tEnd = time.time()
    print('ScreenShot Time = ' + str(float(tEnd - tStart)))
    # im.show()
    return im

def GetScreenPos():
    global timeFlag
    while timeFlag <= 2:
        x,y = pag.position()
        # print('%f, %f' %(x,y))


if __name__ == "__main__":

    thread = threading.Thread(target = FlagController, args = ())
    thread.start()

    # GetScreenPos()
    tStart = time.time()
    screen = CatchScreen(639,6 , 165, 15)
    text = pyocr.image_to_string(screen)
    tEnd = time.time()
    print('Total OCR Time = ' + str(float(tEnd - tStart)))
    print(text)

    # AutoMove()
    
    # while True:  # making a loop
    #     try:  # used try so that if user pressed other than the given key error will not be shown
    #         if keyboard.is_pressed('enter'):  # if key 'enter' is pressed 
    #             AutoMove()
    #             break  # finishing the loop
    #         else:
    #             pass
    #     except:
    #         break  # if user pressed a key other than the given key the loop will break