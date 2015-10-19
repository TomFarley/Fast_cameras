
import numpy as np
import cv2
from create_log import create_log
import matplotlib.pyplot as plt
from copy import deepcopy as copy

class frameHistory:
	"""
	Simple class to store a movie as a history of frames
	
	Frames within a frame history can be accessed by indexing, ie
	
	frames = frameHistory()
	
	frame = frames[1,:,:,:]
	
	
	"""
	
	def __init__(self,N=100):
				
		self.frames = None
		self.N = N	#Store maximum of 100 frames in history by default
		self._updates=0	 #Store the number of times history has been updated
		self.ROI = None
		self.timestamps = []
		self.frameNumbers = []
	def set(self,frames):
		"""
		Set the frame history and do not update
		"""
		
		if type(frames) == type(self):
			self.frames = frames.frames
			self.N = frames.N
			self._updates = frames._updates
			self.timestamps = frames.timestamp
			self.frameNumbers = frames.frameNumbers
		else:	
			self.frames = copy(frames)
			self.N = frames.shape[0]
			self._updates = self.N - 1
		
	def add(self,frame,time=None,frameNo=None):
		"""
		Update the frame history with a new frame.
		If the history already has the desired number of frames, then 
		remove the oldest and add the newest
		
		"""
		if self.frames is None:
			self.frames = frame[np.newaxis,...]
			self._updates += 1
			if time:
				self.timestamps.append(time)
			if frameNo:
				self.frameNumbers.append(frameNo)
		else:
			if self._updates < self.N:
				self.frames = np.append(self.frames,frame[np.newaxis,...],axis=0)
				self._updates += 1
				if time:
					self.timestamps.append(time)
				if frameNo:
					self.frameNumbers.append(frameNo)
			else:
				self._updates += 1
				self.frames[self._updates % self.N] =  frame
				if time:
					self.timestamps[self._updates % self.N] = time
				if frameNo:
					self.frameNumbers[self._updates % self.N] = frameNo
	
		
	
	def append(self,frame):
		"""
		Used to add a frame and increasing framecount in the history
		"""
	
		if self.frames is None:
			self.frames = frame[np.newaxis,...]
			self._updates += 1
		else:
			self.frames = np.append(self.frames,frame[np.newaxis,...],axis=0)
			self._updates += 1
			self.N += 1
			
	
		
	def setROI(self,ROI=[]):
		"""
		sets the ROI of the frames in the history
		
		"""
		
		if not ROI:
			"""
			User selection of the ROI
			
			"""	
			ROI = []
			def onClick(event):
				ROI.append([int(event.ydata),int(event.xdata)])
				plt.scatter([event.xdata],[event.ydata],c='r')
				plt.draw()
			
			fig = plt.figure()
			cid = fig.canvas.mpl_connect('button_press_event', onClick)
			plt.imshow(self[0])
			plt.title("Please select ROI coordinates")
			plt.xlim(0,self[0].shape[1])
			plt.ylim(self[0].shape[0],0)
			plt.show()
			
			print "\n ROI Coordinates: \n",ROI
			
		self.ROI = np.array(ROI)
		
	
	def getROImask(self,ROI=None):
		"""
		return a boolean mask determining which points in the frames are within the ROI
		"""	
		if ROI is None:
			ROI = self.ROI
	
		def point_inside_ROI(point,ROI):

			n = ROI.shape[0]
			
			inside = 0
			x,y = point
			p1x = ROI[0,0]
			p1y = ROI[0,1]
			for i in range(n+1):
				p2x = ROI[i % n,0]
				p2y = ROI[i % n,1]
				if y > min(p1y,p2y) and y <= max(p1y,p2y) and x <= max(p1x,p2x):
					if p1y != p2y:
						xinters = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
					if p1x == p2x or x <= xinters:
						inside = (inside + 1) % 2
					
				p1x,p1y = p2x,p2y
					
			return inside
		
		nx = np.max(ROI[:,0]) - np.min(ROI[:,0]) + 1
		ny = np.max(ROI[-1,1]) - np.min(ROI[0,1]) + 1
		
		
		xpoints = np.arange(nx) + np.min(ROI[:,0])
		ypoints = np.arange(ny) + np.min(ROI[:,1])
		
		pointsinROI = np.zeros(self[0].shape,dtype=np.uint8)
		pointsinROI[...] = False
		for x in xpoints:
			for y in ypoints:
				pointsinROI[x,y] = point_inside_ROI((x,y),ROI)
		
		return np.uint8(pointsinROI)
		
		
	def __iter__(self):
		"""
		Iterate frames using a call like 
		
		for frame in frameHistory:
			
		"""
		for N in np.arange(self.frames.shape[0]):
			yield self.frames[N]
		
		
			
	def __getitem__(self,index,mask=False):
		"""
		Access frames in the frameHistory using
		frameHistory()[i,j,k,...]
		"""
		if not mask:
			return self.frames[index]
		
		
	def __setitem__(self,index,setvalue):
		"""
		Set individual frames using index
		"""
		self.frames[index] = setvalue
		
	
