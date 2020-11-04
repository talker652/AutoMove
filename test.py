#-----------------------------#
#                             #
#   This file for DPCOVPDF.   #
#                             #
#-----------------------------#

import importlib.util
import warnings
warnings.simplefilter("ignore")

# Configparser

from PIL import Image

import os, time
import fitz
import numpy as np
from cv2 import cv2 as cv2
from datetime import datetime

from matplotlib import pyplot as plt
def show(im, islist = False):
	if islist:
		for i in im:
			testimage = Image.fromarray(i)
			testimage.show()
	else :
		testimage = Image.fromarray(im)
		testimage.show()

def showplot(**kwargs):
	datas = kwargs["datas"]
	thrs = kwargs["thrs"]
	title = kwargs["title"]
	axis_x = kwargs["axis_x"]
	axis_y = kwargs["axis_y"]
	plt.plot(datas)
	plt.plot([0, len(datas)], [thrs, thrs], "r", "-")
	plt.title(title, fontsize=18)
	plt.xlabel(axis_x, fontsize=18)
	plt.ylabel(axis_y, fontsize=18)
	plt.show()
	plt.clf()


def toCleanText(text):
	result = ''
	for l in text.split('\n'):
		if l.strip() != '':
			result += l.strip() + '\n'
	return result[0:-1]

def ListToString(l):
	string = ''
	if type(l) == list:
		for text in l:
			string += text + ' '
	else :
		string = l
	return string.strip()



# ## PDF 轉影像 ##
# ---
# + 1. 使用pymupdf套件
# + 2. 轉換後做resize然後輸出陣列
class toImg(object):
	def __init__(self):
		super(toImg, self).__init__()

	def getPage(self, file):
		pdf = fitz.open(file)
		imgs = [self.pdf2img(p) for p in pdf]
		pdf.close()
		return imgs

	def pdf2img(self, pdfPage):
		Mat = fitz.Matrix(8, 8).preRotate(0)
		pix = pdfPage.getPixmap(matrix=Mat, alpha=False)
		h = pix.height
		w = pix.width
		im = Image.frombytes("RGB", [w, h], pix.samples)

		FIX_HEI = 6000
		new_w = int(FIX_HEI/h*w)
		im = im.resize((new_w, FIX_HEI), Image.ANTIALIAS)
		im = np.asarray(im)
		return im


# ## 影像前處理 ##
# ---
# + 1. 二值化 + 黑白反轉
# + 2. 偵測與計算所有區塊的面積，並找到最大者用於校正歪斜
# + 3. 校正影像
# + 4. 使用校正後的影像依照當初設定的「正」的bounding box裁切
# + 5. 再次進行例調整後得到最後目標
# ---
# ```python
# img[::-1] -> upside down
# img[:,::-1] -> horizontal flip
# ```
class dip():
	def gray(self, im):
		return cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

	def binarize(self, im):
		return cv2.threshold(255 - im, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1] #OTSU

	def deNoise(self, im):
		kernel = np.ones([2, 1], dtype=np.uint8)
		dilate = cv2.dilate(im, kernel, interations = 3)
		erode = cv2.erode(dilate, kernel, iterations = 3)
		return erode

	def findBox(self, im):
		contours, hier = cv2.findContours(im, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		c = max(contours, key = cv2.contourArea)
		rect = cv2.minAreaRect(c)
		boxPoints = np.int0(cv2.boxPoints(rect))
		angle = rect[2]
		if angle < -45: angle += 90
		return angle, boxPoints

	def warpAffine(self, im, angle):
		h, w = im.shape[:2]
		center = (w//2, h//2)
		rotMatrix = cv2.getRotationMatrix2D(center, angle, 1.0)
		rotated = cv2.warpAffine(
			im, # 用灰度影像進行校正
			rotMatrix,
			(w, h),
			flags = cv2.INTER_CUBIC,
			borderMode = cv2.BORDER_CONSTANT,
			borderValue = (255, 255, 255)
		) # 型態變換(旋轉)
		return rotMatrix, rotated

	def crop(self, im, bp, m):
		# bp = boxPoints, m = rotMatrix
		pts = np.int64(cv2.transform(np.array([bp]), m))[0]
		y_min = min(pts[0][0], pts[1][0], pts[2][0], pts[3][0])
		y_max = max(pts[0][0], pts[1][0], pts[2][0], pts[3][0])
		x_min = min(pts[0][1], pts[1][1], pts[2][1], pts[3][1])
		x_max = max(pts[0][1], pts[1][1], pts[2][1], pts[3][1])

		# 化負為零
		if y_min < 0: y_min = 0
		if y_max < 0: y_max = 0
		if x_min < 0: x_min = 0
		if x_max < 0: x_max = 0

		# 裁切影像 (用校正後的灰度影像根據當初「正」的框來裁切)
		crop = im[x_min:x_max, y_min:y_max]

		#計算縮放比例 -> shape [5600, None, channel] (cv2 format)
		FIX_HEI = 5600
		baseHEI, baseWID = crop.shape[:2]
		rate = baseHEI / baseWID
		newW = int(FIX_HEI / rate)
		crop = Image.fromarray(crop)
		# 用PIl resize效果比較好
		crop = crop.resize((newW, FIX_HEI), Image.ANTIALIAS)

		# 轉回 cv2 影像
		crop = np.asarray(crop)
		return crop

	def sharpen(self, im):
		color    = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
		alpha    = 0.5
		beta     = 0
		adjusted = cv2.convertScaleAbs(color, alpha=alpha, beta=beta)

		kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
		dst    = cv2.filter2D(adjusted, -1, kernel=kernel)
		g      = cv2.cvtColor(dst, cv2.COLOR_RGB2GRAY)
		return g

	def projection_horizontal(self, im, rt = 0.25, cutEdge = True, show_ = False, getLast = False, retFlag = False, retImg = np.array((0,0))):
		bins = self.binarize(im)
		row_sum = np.sum(255-bins, axis = 1).tolist()

		thrs = int(max(row_sum)*rt)
		if show_:
			showplot(
				datas=row_sum,
				thrs=thrs,
				title="Horizontal Projection",
				axis_x="width",
				axis_y="Sum"
			)

		row_section = []
		tempBox = [0, 0]
		for k, v in enumerate(row_sum):
			if v > thrs and tempBox[0] == 0:
				tempBox[0] = 0 if k < 0 else k
			elif v <= thrs and tempBox[0] != 0:
				tempBox[1] = k
				row_section.append(tempBox)
				tempBox = [0, 0]
		if getLast:
			row_section.append([tempBox[0], im.shape[0]])
		if retFlag != False:
			if cutEdge:
				rowSet = [retImg[k[0]+3:k[1]-2, 0:retImg.shape[1]] for k in row_section]
			else:
				rowSet = [retImg[k[0]:k[1], 0:retImg.shape[1]] for k in row_section]
		else :
			if cutEdge:
				rowSet = [im[k[0]+3:k[1]-2, 0:im.shape[1]] for k in row_section]
			else:
				rowSet = [im[k[0]:k[1], 0:im.shape[1]] for k in row_section]
		return row_section, rowSet

	def projection_horizontal_white(self, im, thrsflag = False, thrsmean = 20, rt = 0.1, cutEdge = True, show_ = False):
		bins = cv2.threshold(255 - im, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
		row_sum = np.sum(255 - bins, axis = 1).tolist()
		row_mean = np.mean(row_sum)
		# show(bins,'test')
		# print(row_mean)
		if thrsflag == True :
			thrs = row_mean/thrsmean
		else :
			thrs = 0
		# print(thrs)
		if show_:
			showplot(
				datas=row_sum,
				thrs=thrs,
				title="Horizontal White Projection",
				axis_x="width",
				axis_y="Sum"
			)

		row_section = []
		tempBox = [0, 0]
		for k, v in enumerate(row_sum):
			if v > thrs and tempBox[0] == 0:
				tempBox[0] = 0 if k < 0 else k
			elif v <= thrs and tempBox[0] != 0:
				tempBox[1] = k
				if tempBox[1] - tempBox[0] >= 30:
					row_section.append(tempBox)
				tempBox = [0, 0]
		# if row_sum[0] > 0:
		# 	row_section.remove(row_section[0])
		rowSet = []
		if cutEdge or thrsflag:
			for k in row_section:
				k0 = k[0] - 5 if k[0] >= 5 else 0
				k1 = k[1] + 5 if k[1] + 5 < im.shape[0] else im.shape[0]
				rowSet.append(im[k0:k1, 0:im.shape[1]])
		else:
			rowSet = [im[k[0]:k[1], 0:im.shape[1]] for k in row_section]
		return row_section, rowSet

	def projection_vertical(self, im, rt = 0.25, cutEdge = True, show_ = False, retFlag = False ,retImg = np.array((0,0))):
		bins = self.binarize(im)
		col_sum = np.sum(255-bins, axis = 0).tolist()
		

		thrs = int(max(col_sum)*rt)
		if show_:
			showplot(
				datas=col_sum,
				thrs=thrs,
				title="Vertical Projection",
				axis_x="width",
				axis_y="Sum"
			)

		col_section = []
		tempBox = [0, 0]
		for k, v in enumerate(col_sum):
			if v > thrs and tempBox[0] == 0:
				tempBox[0] = 1 if k < 0 else k
			elif v <= thrs and tempBox[0] != 0:
				tempBox[1] = k

				col_section.append(tempBox)
				tempBox = [0, 0]
		if retFlag == False:
			if cutEdge:
				colSet = [im[0:im.shape[0], k[0]+10:k[1]-10] for k in col_section]
			else:
				colSet = [im[0:im.shape[0], k[0]:k[1]] for k in col_section]
		else :
			if cutEdge:
				colSet = [retImg[0:retImg.shape[0], k[0]+10:k[1]-10] for k in col_section]
			else:
				colSet = [retImg[0:retImg.shape[0], k[0]:k[1]] for k in col_section]
		return col_section, colSet

	def projection_vertical_white(self, im, thrsflag = False, thrsmean = 5, erodeFlag = True, rt = 0.25, cutEdge = True, show_ = False):
		# bins = self.binarize(im)
		# col_sum = np.sum(255-bins, axis = 0).tolist()
		if erodeFlag :
			image = self.removeLine(im) 
			kernel = np.ones((4,30),np.uint8)
			image_erode = cv2.erode(image, kernel,iterations = 3)
		else :
			image_erode = im
		
		bins = cv2.threshold(255 - image_erode, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
		col_sum = np.sum(255 - bins, axis = 0).tolist()


		col_mean = np.mean(col_sum)

		if thrsflag == True :
			thrs = col_mean/thrsmean
		else :
			thrs = 0

		if show_:
			showplot(
				datas=col_sum,
				thrs=thrs,
				title="Vertical White Projection",
				axis_x="width",
				axis_y="Sum"
			)

		col_section = []
		tempBox = [0, 0]
		for k, v in enumerate(col_sum):
			if v > thrs and tempBox[0] == 0:
				tempBox[0] = 1 if k < 0 else k
			elif v <= thrs and tempBox[0] != 0:
				tempBox[1] = k
				if tempBox[1] - tempBox[0] >= 10:
					col_section.append(tempBox)
				tempBox = [0, 0]
		if cutEdge:
			colSet = [im[0:im.shape[0], k[0]+10:k[1]-10] for k in col_section]
		else:
			colSet = [im[0:im.shape[0], k[0]:k[1]] for k in col_section]
		return col_section, colSet

	def projection_spe(self, im, rv = False, show_ = False):
		_, bins = cv2.threshold(im, 127, 255, cv2.THRESH_BINARY)
		col_sum = np.sum(bins, axis = 0).tolist()

		if show_:
			showplot(
				datas=col_sum,
				thrs=0,
				title="Special Projection",
				axis_x="width",
				axis_y="Sum"
			)

		if rv: col_sum = col_sum[::-1]
		col_section = []
		tempBox = [0, 0]

		for k, v in enumerate(col_sum):
			if v > 0 and tempBox[0] == 0:
				tempBox[0] = 1 if k - 1 < 0 else k - 5

			elif v <= int(min(col_sum)) and tempBox[0] != 0:
				tempBox[1] = k if k + 1 < 0 else k + 5
				col_section.append(tempBox)
				tempBox = [0, 0]

			elif k+1 == len(col_sum) and tempBox[0] != 0 and tempBox[1] == 0:
				tempBox[1] = k if k + 1 < 0 else k + 5
				col_section.append(tempBox)

		return col_section, [im[:im.shape[0], k[0]:k[1]] for k in col_section]

	def cutSpace(self, im, show = False):
		def getPos(m):
			k = 0
			thrs = int(max(m)*0.075)
			for key, val in enumerate(m):
				if val > thrs:
					k = key
					break
			return k

		_, w = im.shape[:2]
		col_sum = np.sum(self.binarize(im), axis = 0).tolist()

		if show:
			showplot(
				datas=col_sum,
				thrs=0,
				title="cutSpace",
				axis_x="Width",
				axis_y="Sum"
			)

		head = getPos(col_sum)
		tail = len(col_sum) - getPos(col_sum[::-1])

		# 放寬頭尾
		head = 0 if head - 10 < 0 else head - 10
		tail = w if tail + 10 > w else tail + 10

		nim = im[::, head:tail]
		return nim

	def bigCanvas(self, im):
		h, w = im.shape[:2]
		canvas = np.zeros((h+10, w+20) ,np.uint8) + 255
		canvas[5:h+5, 15:w+15] = im
		return canvas

	def hasValue(self, im):
		h, _ = im.shape[:2]
		cut3 = h//3 # 橫切三等份
		ROI = im[cut3:cut3*2, ::] # 取中間段
		avg = np.mean(ROI)

		rst = True if avg%int(avg) != 0 else False
		return rst

	def TPLMatching(self, im = None, tpl = None, thrs = 0.8, method = cv2.TM_CCOEFF_NORMED):
		def rmSame(pts, thrs):
			elements = []
			for x, y in pts:
				for ele in elements:
					if ((x-ele[0])**2 + (y-ele[1])**2) < thrs**2: break
				else:
					elements.append((x,y))
			return elements[0]

		matchResult = cv2.matchTemplate(im, tpl, method)
		loc = np.where(matchResult >= thrs)
		tplH, tplW = tpl.shape[:2]
		x1, y1 = rmSame(zip(*loc[::-1]), min(tplH, tplW))
		x2, y2 = x1 + tplW, y1 + tplH
		return [(x1, y1), (x2, y2)]

	def resize(self, im, rt):
		h, w = im.shape[:2]

		if h > w:
			r = w / h
			h = h * rt
			w = h * r
		elif w > h:
			r = h / w
			w = w * rt
			h = w * r
		else:
			w, h = np.asarray([w, h]) * rt

		nim = cv2.resize(im, (int(w), int(h)), interpolation = cv2.INTER_CUBIC)
		return nim

	def removeLine(self, img, lineLenth = 1000):
		edges = cv2.Canny(img, 50, 150, apertureSize=3)  
		lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 1000, minLineLength=lineLenth,maxLineGap=10)  
		try:	
			for line in lines:
				x1,y1,x2,y2 = line[0]
				cv2.line(img,(x1,y1),(x2,y2),(255,255,255),20)
			return img
		except:
			return img

# ## 剖析資料 ##
# ---
# + 1. 從stat_ocr取得的OCR結果進行整理
# + 2. 每個function處理各自的資料
# + 3. 每個item自成一支function處理
from datetime import datetime
import re
class retriever():
	def __init__(self, datas):
		self.datas = datas

	def MV(self):
		try:
			ret = []
			for key in self.datas.keys():
				line =  self.datas[key][1].replace('\n', ' ').replace('_', '')
				MV_pos = line.find('MV')
				upP_pos = line.find('(')
				if upP_pos > MV_pos :
					result = line[MV_pos+2:upP_pos]
				else :
					result = 'no recg'
				result = result.strip()
				ret.append(result)
			return ret[0]
		except Exception as e :
			print('[LOD_filter-retriever]' + str(e))
			return 'pass' if self.datas[key] == 'pass' else 'err'

	def FV(self):
		try:
			ret = []
			for key in self.datas.keys():
				line =  self.datas[key][1].replace('\n', ' ').replace('_', '')
				FV_pos = line.find('FV')
				in_pos = line.find('in')
				if in_pos > FV_pos :
					result = line[FV_pos+2:in_pos]
				else :
					result = 'no recg'
				result = result.strip()
				ret.append(result)
			return ret[0]
		except Exception as e :
			print('[LOD_filter-retriever]' + str(e))
			return 'pass' if self.datas[key] == 'pass' else 'err'
			
	def PortName(self):
		try:
			ret = []
			for key in self.datas.keys():
				line =  self.datas[key][1].replace('\n', ' ').replace('_', '')
				in_pos = line.find('in')
				port_pos = line.find('port')
				if port_pos > in_pos :
					result = line[in_pos+2:port_pos]
				else :
					result = 'no recg'
				result = result.strip()
				ret.append(result)
			return ret[0]
		except Exception as e :
			print('[LOD_filter-retriever]' + str(e))
			return 'pass' if self.datas[key] == 'pass' else 'err'

	def StartDate(self):
		try:
			ret = []
			for key in self.datas.keys():
				line =  self.datas[key][1].replace('\n', ' ').replace('_', '')[-32:]
				from_pos = line.find('from')
				to_pos = line.find('to')
				if to_pos > from_pos and to_pos != -1 and from_pos != -1:
					string = line[from_pos+4:to_pos]
					string = ''.join([x for x in string if x.isdigit()])
					try:
						date = datetime.strptime(string, "%d%m%Y")
					except :
						date = datetime.strptime(string, "%Y%m%d")
					result = datetime.strftime(date, "%d-%b-%y")
				else :
					result = 'no recg'
				result = result.strip()
				ret.append(result)
			return ret[0]
		except Exception as e :
			print('[LOD_filter-retriever]' + str(e))
			return 'pass' if self.datas[key] == 'pass' else 'err'

	def EndDate(self):
		try:
			ret = []
			for key in self.datas.keys():
				line =  self.datas[key][1].replace('\n', ' ').replace('_', '')[-32:]
				to_pos = line.find('to')
				if len(line) > to_pos and to_pos != -1:
					string = line[to_pos+2:]
					string = ''.join([x for x in string if x.isdigit()])
					try:
						date = datetime.strptime(string, "%d%m%Y")
					except :
						date = datetime.strptime(string, "%Y%m%d")
					result = datetime.strftime(date, "%d-%b-%y")
				else :
					result = 'no recg'
				result = result.strip()
				ret.append(result)
			return ret[0]
		except Exception as e :
			print('[LOD_filter-retriever]' + str(e))
			return 'pass' if self.datas[key] == 'pass' else 'err'
	
	def SKJ(self):
		try:
			ret = []
			for key in self.datas.keys():
				line =  self.datas[key][2].replace('\n', '').replace(' ','')
				SKJ_pos = line.find('SKJ')
				YF_pos = line.find('YF')
				if SKJ_pos < YF_pos :
					string = line[SKJ_pos+3:YF_pos].replace('I','1')
					result = ''.join([x for x in string if x.isdigit() or x == '.'])
				else :
					result = 'no recg'
				result = result.strip()
				ret.append(result)
			return ret[0]
		except Exception as e :
			print('[LOD_filter-retriever]' + str(e))
			return 'pass' if self.datas[key] == 'pass' else 'err'

	def YFN(self):
		try:
			ret = []
			for key in self.datas.keys():
				line =  self.datas[key][2].replace('\n', '').replace(' ','')
				YF_pos = line.find('YF')
				BE_pos = line.find('BE')
				if YF_pos < BE_pos :
					string = line[YF_pos+2:BE_pos].replace('I','1')
					result = ''.join([x for x in string if x.isdigit() or x == '.'])
				else :
					result = 'no recg'
				result = result.strip()
				ret.append(result)
			return ret[0]
		except Exception as e :
			print('[LOD_filter-retriever]' + str(e))
			return 'pass' if self.datas[key] == 'pass' else 'err'

	def BIG(self):
		try:
			ret = []
			for key in self.datas.keys():
				line =  self.datas[key][2].replace('\n', '').replace(' ','')
				BE_pos = line.find('BE')
				if BE_pos < len(line) :
					string = line[BE_pos+2:].replace('I','1')
					result = ''.join([x for x in string if x.isdigit() or x == '.'])
				else :
					result = 'no recg'
				result = result.strip()
				ret.append(result)
			return ret[0]
		except Exception as e :
			print('[LOD_filter-retriever]' + str(e))
			return 'pass' if self.datas[key] == 'pass' else 'err'

	def GrandTotal(self):
		try:
			ret = []
			for key in self.datas.keys():
				line =  self.datas[key][3].replace('\n', '').replace(' ','')
				result = ''.join([x for x in line if x.isdigit() or x == '.'])
				if len(result) > 0 :
					pass
				else :
					result = 'no recg'
				result = result.strip()
				ret.append(result)
			return ret[0]
		except Exception as e :
			print('[LOD_filter-retriever]' + str(e))
			return 'pass' if self.datas[key] == 'pass' else 'err'

	

# ## OCR辨識(tesseract版本) ##
# ---
# + 1. 取得位置
# + 2. OCR
# + 3. filte data
# + 4. return pure data
import requests, json
import pytesseract as pyocr
from pytesseract import Output as tessOut
class ocr(dip):
	def parsingTemplate(self, im):
		boxImage = []
		try:
			im = cv2.fastNlMeansDenoising(im,None,10,7,21)
			kernel = np.ones((35,3),np.uint8)
			image_erode = cv2.erode(im, kernel,iterations = 3)
			h_cuts, _ = self.projection_horizontal_white(image_erode, thrsflag=True, thrsmean=10) 
			# show(h_imgs, True)
			for h_cut in h_cuts:
				h_img = im[h_cut[0]:h_cut[1], :]
				# show(h_img)
				boxImage.append(h_img)
			return boxImage

		except Exception as e:
			print('[LOD_filter-ocr-parseTemplate]' + str(e))
			return boxImage

	def recg(self, im, lang="eng", cfg = False):
		if cfg:
			df = pyocr.image_to_data(im, lang = lang, config = cfg, output_type = tessOut.DATAFRAME).dropna()
		else:
			df = pyocr.image_to_data(im, lang = lang, output_type = tessOut.DATAFRAME).dropna()
		return df

# ## 主程式 ##
# ---
# + 1. 繼承toImg, dip, ocr三支Class
# + 2. 通常由start_ocr先做，影像前處理後做ocr取得34列結果
# + 3. 結果送到retriever處理
# + 4. 最後storageDB處理

class main(toImg, ocr):
	__version__ = "Dev.2.0.0"

	def __init__(self):
		super(main, self).__init__()

	def start_ocr(self, filePath):
		
		dataImgs = {}
		datas = retriever(dataImgs)
		page = self.getPage(filePath)
		try:
			for k, pg in enumerate(page):
				try:
					if pg.shape[0] < pg.shape[1] :
						pg = cv2.rotate(page, cv2.ROTATE_90_CLOCKWISE)
					try :
						im = self.gray(pg)
					except:
						im = pg
					# 降噪 依據情況使用
					# im = cv2.fastNlMeansDenoising(im,None,10,7,21)
					#將大圖切成小圖 每張小圖都代表一個文字或一部分關鍵版面 
					textImgs = self.parsingTemplate(im)
					ocr_rst = []
					if textImgs:
						for textImg in textImgs:
							df = pyocr.image_to_string(textImg, lang='eng')
							df = toCleanText(df)
							# print(df)
							if len(df.strip()) != 0 :
								ocr_rst.append(df.strip())
							else :
								# 沒辨識出文字 再切一刀 避免因為底線的影響
								_, subimgs = self.projection_horizontal_white(textImg)
								for subimg in subimgs:
									if subimg.shape[0] > textImg.shape[0] * 0.5:
										df = pyocr.image_to_string(subimg)
										if len(df.strip()) != 0:
											ocr_rst.append(df.strip())
										else :
											hasText = self.hasValue(subimg)
											df = 'no recg' if hasText else 'no value'
											ocr_rst.append(df.strip())
					else:
						# 沒有正確切出圖片 格式太差
						ocr_rst.append("pass")
					dataImgs[k] = ocr_rst
				except Exception as e:
					print('[LOD_filter-start_ocr-dataImgPass] ' + str(e))
					dataImgs[k] = ocr_rst
			datas = retriever(dataImgs)
			return [True, datas]
		except Exception as e:
			print('[LOD_filter-start_ocr] '+str(e))
			return [False, datas]

if __name__ == "__main__":
	# filePath = "./IFIMS-_LOGSHEET_GAR-442MT.pdf"
	filePath = "./MR06 - FONG KUO 866 - LOD.pdf"
	rt = main().start_ocr(filePath)
	# pass		為文件格式太差 沒有辨識
	# err		有辨識 但是發生錯誤 通常發生在日期
	# no value	此格為空 Vessel name/call sign沒切開也會是no value
	# no recg	此格無法辨識 
	print(rt[1].MV())
	print(rt[1].FV())
	print(rt[1].PortName())
	print(rt[1].StartDate())
	print(rt[1].EndDate())
	print(rt[1].SKJ())
	print(rt[1].YFN())
	print(rt[1].BIG())
	print(rt[1].GrandTotal())

