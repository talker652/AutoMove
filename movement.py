# https://blog.xuite.net/jiehui_prompt/Heal/413343940-VB+%E9%8D%B5%E7%9B%A4%E6%A8%A1%E6%93%AC
# http://blog.itpub.net/26736162/viewspace-2644877/

import os
import time
import win32gui
import win32api
import win32con

def click(x,y): #第一種
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,x,y,0,0)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,x,y,0,0)

def move(dx, dy):
    win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, int(dx), int(dy), 0, 0)

def testmove():

    x = 200
    y = 200
    duration = 1 * 100
    dx = x/duration
    dy = y/duration

    for i in range(100):
        time.sleep(0.01)
        move(dx,dy)
    click(x,y)

    win32api.keybd_event(0x53, win32api.MapVirtualKey(0x53,0) , 0, 0)
    time.sleep(0.05)
    win32api.keybd_event(0x53, win32api.MapVirtualKey(0x53,0) , win32con.KEYEVENTF_KEYUP, 0)