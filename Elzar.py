"""
Class to manually track filaments in a given movie
"""

from readMovie import readMovie
from fieldlineTracer import get_fieldline_tracer
from backgroundSubtractor import *
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor,Button,Slider,CheckButtons
from matplotlib.text import Annotation as annotate
from matplotlib.patches import Rectangle
from matplotlib import cm
import matplotlib.colors as colors
from utils import *
import numpy as np
from copy import deepcopy as copy
import cv2
from scipy.interpolate import interp2d
from tkFileDialog import asksaveasfile
from scipy.stats import moment,describe
from scipy.signal import argrelmax,argrelmin
try:
	import cpickle as pickle
except:
	import pickle

try:
	import Fitting,MachineGeometry,Render
except:
	print("WARNING: No Camtools module found. Needed for calibration.")


class Elzar(object):
	
	def __init__(self,Nframes=None,startpos=0.0,gfile=None,moviefile=None,calibfile=None,machine='MAST',shot=None,time=0.3):
		
		if shot is not None:
			self.frames = readMovie(str(shot),Nframes=Nframes,startpos=startpos)
			print("Finished reading movie file")
			
			try:
				self.calibration = Fitting.CalibResults(str(shot))
				self.DISABLE_PROJECT = False
			except:
				print("WARNING: Unable to read calibration file. Camera projection disabled.")
				self.DISABLE_PROJECT = True
			
			self.tracer = get_fieldline_tracer(type='RK4',machine='MAST',shot=shot,time=time,interp='linear')
			self.DISABLE_TRACER = False

		else:
			if gfile is None:
				gfile = raw_input("gfile: ")
			try:				
				self.tracer = get_fieldline_tracer(type='RK4',gfile=gfile,time=time,interp='linear')
				self.DISABLE_TRACER = False
			except:
				print "WARNING: Unable to load gfile "+gfile+"! Field line tracing disabled."
				self.DISABLE_TRACER = True
			if moviefile is None:
				moviefile = raw_input("Movie file or shot number: ")	
			print("Reading Movie file, please be patient")				
			self.frames = readMovie(moviefile,Nframes=Nframes,startpos=startpos)
			print("Finished reading movie file")

			if calibfile is None:
				calibfile = raw_input("Calibration file: ")
			try:
				self.calibration = Fitting.CalibResults(calibfile)
				self.DISABLE_PROJECT = False
			except:
				print("WARNING: Unable to read calibration file. Camera projection disabled.")
				self.DISABLE_PROJECT = True

		self._gfile = gfile
		self.flines = []
		self.projectLines = []
					
		self._currentframeNum = 0
		self._currentframeData = self.frames[self._currentframeNum]
		self._currentframeDisplay = cv2.cvtColor(self._currentframeData,cv2.COLOR_GRAY2BGR)	
		self._currentframeTime = self.frames.timestamps[self._currentframeNum]
		self._currentframeMark = self.frames.frameNumbers[self._currentframeNum]
				
		self.bgsub = backgroundSubtractorMin(20)
		#Initialize background model
		for frame in self.frames[0:19]:
			dummy = self.bgsub.apply(frame)
		
		self.widgets = {}
		self.dataCursor = None
	
		try:
			self.CADMod = MachineGeometry.MAST('high')
			self.wireframe = Render.MakeRender(self.CADMod,self.calibration,Verbose=False,Edges=True,EdgeWidth=1,EdgeMethod='simple')
			self.DISABLE_CCHECK = False
		except:
			print("WARNING: No CAD model found for MAST. Disabling calibration checking.")
			self.DISABLE_CCHECK = True
		
		self._currentZ = 0.0
		self._currentR = 1.45
		self._currentphi = 0.0

                self.gammaEnhance = False
                self.applySub = False
                self.threshold = False
                self.histEq = False
                self.edgeDet = False
                self.noiseRmv = False
                self.negative = False
                self.gamma = 1.0


	def runUI(self):
		
		#Initialize some parameters for the UI
		self.gammaEnhance = False
		self.applySub = False
		self.threshold = False
		self.histEq = False
		self.edgeDet = False
		self.noiseRmv = False
		self.negative = False
		self.gamma = 1.0
		self.fieldlineArtists = []
		self.flineTxt = None
		self.selectedPixel = None
		self.pixelplot = None
		self.selectedLine = None
		self.linePlot = None
		axcolor = 'lightgoldenrodyellow'
		self.mask = copy(self._currentframeDisplay)
		self.mask[...]	= 1
		self.mask = np.uint8(self.mask)
		self.wireframeon = None
		
		#Set up UI window
		fig = plt.figure(figsize=(8,8),facecolor='w',edgecolor='k')
		#Set up axes for displaying images
		frameax = plt.axes([0.0,0.30,0.6,0.6])
		frame = self.enhanceFrame(copy(self._currentframeData))
		self.img = frameax.imshow(frame)
		frameax.set_axis_off()
		frameax.set_xlim(0,frame.shape[1])
		frameax.set_ylim(frame.shape[0],0)
		text = 'Frame: '+str(self._currentframeMark)+'   Time: '+str(self._currentframeTime)+' [ms]'
		self.frametxt = frameax.annotate(text,xy=(0.05,0.95),xycoords='axes fraction',color='white',fontsize=8)
		frameax.add_artist(self.frametxt)
		
		#Set up axis for equilibrium plot
		eqax = plt.axes([0.7,0.25,0.25,0.5])
		eqax.set_xlim(0.0,2.0)
		eqax.set_ylim(-2.0,2.0)
		eqax.set_title('Poloidal Cross-section')
		eqax.set_ylabel('Z (m)')
		eqax.set_xlabel('R (m)')
		if not self.DISABLE_TRACER: wallplot = eqax.plot(self.tracer.eq.wall['R'],self.tracer.eq.wall['Z'],'-k')
		
		#Image enhancement selector
		enhancelabels = ('BG subtraction','Threshold','Gamma enhance','Detect edges','Equalise','Reduce Noise','Negative')
		enhanceCheck = CheckButtons(plt.axes([0.7,0.05,0.25,0.16]),enhancelabels,(False,False,False,False,False,False,False))
		gammaSlide = Slider(plt.axes([0.75,0.02,0.2,0.02],axisbg=axcolor), 'Gamma', 0.0, 3.0, valinit=1.0 )
		self._enhancedFrame = self._currentframeDisplay		
		def setEnhancement(label):
			if label == 'BG subtraction': self.applySub = not self.applySub
			elif label == 'Threshold' : self.threshold = not self.threshold
			elif label == 'Gamma enhance' : self.gammaEnhance = not self.gammaEnhance
			elif label == 'Detect edges'  : self.edgeDet = not self.edgeDet
			elif label == 'Equalise' : self.histEq = not self.histEq
			elif label == 'Reduce Noise' : self.noiseRmv = not self.noiseRmv
			elif label == 'Negative' : self.negative = not self.negative
			self._enhancedFrame = self.mask*self.enhanceFrame(self._currentframeData)
		enhanceCheck.on_clicked(setEnhancement)
		
		def updateGamma(val):
			self.gamma = val
		gammaSlide.on_changed(updateGamma)
		
		#Field line launching area
		axR = plt.axes([0.2, 0.1, 0.35, 0.02], axisbg=axcolor)
		Rslide = Slider(axR, '$R$', 0.2, 2.0, valinit = 1.41 )
		self._currentR = 1.41	
		
		def updateR(val):
			self._currentR = val
		Rslide.on_changed(updateR)
		
		axZ = plt.axes([0.2, 0.06, 0.35, 0.02], axisbg=axcolor)
		Zslide = Slider(axZ, '$Z$', -2.0, 2.0, valinit = 0.0 )
		
		
		def updateZ(val):
			self._currentZ = val
		Zslide.on_changed(updateZ)
		
		axPhi = plt.axes([0.2,0.02,0.35,0.02],  axisbg=axcolor)
		PhiSlide = Slider(axPhi, '$\phi$', 0, 360, valinit = 180 )
		self._currentphi = 60.0
		
		def updatePhi(val):
			if not self.DISABLE_TRACER:
				dphi = val - self._currentphi
				if not self.flines:
					self._currentphi = val
					return
					
				ang = 2.0*np.pi*dphi/360.0
				self.flines[-1].rotate_toroidal(ang)
				linepoints = self.projectFieldline(self.flines[-1])
				self.projectLines[-1] = linepoints
				self.fieldlineArtists[-1].set_data(linepoints[:,0],linepoints[:,1])
				fig.canvas.draw()
				self._currentphi = val 
				self.flineTxt.set_visible(False)
				self.flineTxt = frameax.annotate("Summed intensity: %.2f" % (self.sumIntensity(linepoints)),xy=(0.6,0.01),
				xycoords='axes fraction',color='white',fontsize=8)
			
		PhiSlide.on_changed(updatePhi)
		
		launchButton = Button(plt.axes([0.01,0.08,0.16,0.05]),'Launch',hovercolor='r')
		def onpick(event,annotation):
			if not self.DISABLE_TRACER:
				thisline = event.artist
				thisind = self.fieldlineArtists.index(thisline)
				thisfline = self.flines[thisind]
				x,y = annotation.xy
				lineind = findNearest(self.projectLines[thisind],(x,y))
				R = thisfline.R[lineind]
				Z = thisfline.Z[lineind]
				psiN = self.tracer.eq.psiN(R,Z)
				phi = (thisfline.phi[lineind]*360.0/(2*np.pi)) % 360.0 
				annotation.set_text(self.dataCursor.template % (R,Z,phi,psiN))
				annotation.set_visible(True)
				event.canvas.draw()
			
		def launchFieldline(event):
			if not self.DISABLE_TRACER:
				self.flines.append(self.tracer.trace(self._currentR,self._currentZ,phistart = 2.0*np.pi*self._currentphi/360.0,mxstep=1000,ds=0.05))	
				linepoints = self.projectFieldline(self.flines[-1])
				self.projectLines.append(linepoints)
				flineplot, = frameax.plot(linepoints[:,0],linepoints[:,1],picker=5,lw=1.5)
				self.fieldlineArtists.append(flineplot)
				self.eqlineplot, = eqax.plot(self.flines[-1].R,self.flines[-1].Z)
				if self.dataCursor:
					self.dataCursor.clear(fig)
					self.dataCursor.disconnect(fig)
				self.dataCursor = DataCursor(self.fieldlineArtists,func=onpick,template="R: %.2f\nZ: %.2f\nphi: %.2f\npsiN: %.2f")	
				self.dataCursor.connect(fig)	
				if self.flineTxt:
					self.flineTxt.set_visible(False)
				self.flineTxt = frameax.annotate("Summed intensity: %.2f" % (self.sumIntensity(linepoints)),xy=(0.6,0.01),
				xycoords='axes fraction',color='white',fontsize=8)
			else:
				print("WARNING: Cannot launch field line, tracer is disabled!")
		launchButton.on_clicked(launchFieldline)
					
		def onRelease(event):
			if self.dataCursor:
				self.dataCursor.clear(fig)
		fig.canvas.mpl_connect('button_release_event', onRelease)	
		
		clearButton = Button(plt.axes([0.01,0.02,0.16,0.05]),'Clear',hovercolor='r')
		
		def clearFieldlines(event):
			frameax.clear()
			frame = self.mask*self.enhanceFrame(copy(self._currentframeData))
			self.img = frameax.imshow(frame)
			text = 'Frame: '+str(self._currentframeMark)+'   Time: '+str(self._currentframeTime)+' [ms]'
			self.frametxt = frameax.annotate(text,xy=(0.05,0.95),xycoords='axes fraction',color='white',fontsize=8)
			frameax.set_axis_off()
			frameax.set_xlim(0,frame.shape[1])
			frameax.set_ylim(frame.shape[0],0)
			eqax.clear()
			eqax.set_xlim(0.0,2.0)
			eqax.set_ylim(-2.0,2.0)
			eqax.set_title('Poloidal Cross-section')
			eqax.set_ylabel('Z (m)')
			eqax.set_xlabel('R (m)')
			if not self.DISABLE_TRACER: wallplot = eqax.plot(self.tracer.eq.wall['R'],self.tracer.eq.wall['Z'],'-k')
			self.flines = []
			self.fieldlineArtists = []
			self.projectLines = []
			self.dataCursor = None
			#self.flineTxt.set_visible(False)
			self.flineTxt = None
			self.selectedLine = None
		clearButton.on_clicked(clearFieldlines)
		
		#Frame selection section
		
		nextButton =  Button(plt.axes([0.01,0.92,0.13,0.05]),'Next',hovercolor='r')
		
		def plotNext(event):
			self.nextFrame()
			frame = self.mask*self.enhanceFrame(self._currentframeData)
			self.img.set_data(frame)
			text = 'Frame: '+str(self._currentframeMark)+'   Time: '+str(self._currentframeTime)+' [ms]'
			self.frametxt.set_visible(False)
			self.frametxt = frameax.annotate(text,xy=(0.05,0.95),xycoords='axes fraction',color='white',fontsize=8)
			frameax.add_artist(self.frametxt)
			if self.flineTxt:
				self.flineTxt.set_visible(False)
				self.flineTxt = frameax.annotate("Summed intensity: %.2f" % (self.sumIntensity(self.projectLines[-1])),xy=(0.6,0.01),
				xycoords='axes fraction',color='white',fontsize=8)
			fig.canvas.draw()
		nextButton.on_clicked(plotNext)
		
		prevButton =  Button(plt.axes([0.15,0.92,0.13,0.05]),'Previous',hovercolor='r')
		
		def plotPrev(event):
			self.previousFrame()
			frame = self.mask*self.enhanceFrame(copy(self._currentframeData))
			self.img.set_data(frame)
			text = 'Frame: '+str(self._currentframeMark)+'   Time: '+str(self._currentframeTime)+' [ms]'
			self.frametxt.set_visible(False)
			self.frametxt = frameax.annotate(text,xy=(0.05,0.95),xycoords='axes fraction',color='white',fontsize=8)
			frameax.add_artist(self.frametxt)
			if self.flineTxt:
				self.flineTxt.set_visible(False)
				self.flineTxt = frameax.annotate("Summed intensity: %.2f" % (self.sumIntensity(self.projectLines[-1])),xy=(0.6,0.01),
				xycoords='axes fraction',color='white',fontsize=8)
			fig.canvas.draw()
		prevButton.on_clicked(plotPrev)
			
		refreshButton = Button(plt.axes([0.29,0.92,0.13,0.05]),'Refresh',hovercolor='r')
		
		def refreshPlot(event):
			self.mask = copy(self._currentframeDisplay)
			self.mask[...] = 1
			self.mask = np.uint8(self.mask)
			frame = self.mask*self.enhanceFrame(copy(self._currentframeData))
			self.img.set_data(frame)
			if self.flineTxt:
				self.flineTxt.set_visible(False)
				self.flineTxt = frameax.annotate("Summed intensity: %.2f" % (self.sumIntensity(self.projectLines[-1])),xy=(0.6,0.01),
				xycoords='axes fraction',color='white',fontsize=8)
			fig.canvas.draw()
			self._enhancedFrame = self.mask*self.enhanceFrame(self._currentframeData)	
		refreshButton.on_clicked(refreshPlot)

		saveButton = Button(plt.axes([0.43,0.92,0.13,0.05]),'Save',hovercolor='r')
		
		def saveFrame(event):
			"""
			Save both current plot in frame axis, and save image data as pngs
			"""
			savefile = raw_input("Safe file: ")
			extent = frameax.get_position().get_points()
			inches = fig.get_size_inches()
			extent *= inches
			bbox = frameax.get_position()
			bbox.set_points(extent)
			if savefile != '':
				fig.savefig(savefile,bbox_inches = bbox)
				savefileparts = savefile.split('.')
				plt.imsave(savefileparts[0]+"DATA.png",self._enhancedFrame)
		saveButton.on_clicked(saveFrame)
		
		ROIButton = Button(plt.axes([0.72,0.81,0.22,0.05]),'Set ROI',hovercolor='r')
		
		self.selector = ROISelector(self.img)
		def setROI(event):
			self.ROI = np.asarray(self.selector.coords)
			self.mask[self.ROI[0,1]:self.ROI[1,1],self.ROI[0,0]:self.ROI[1,0]] = int(0)
			self.mask = int(1) - self.mask
			frame = self.mask*self.enhanceFrame(copy(self._currentframeData))
			self.img.set_data(frame)
			if self.flineTxt:
				self.flineTxt.set_visible(False)
				self.flineTxt = frameax.annotate("Summed intensity: %.2f" % (self.sumIntensity(self.projectLines[-1])),xy=(0.6,0.01),
				xycoords='axes fraction',color='white',fontsize=8)
			fig.canvas.draw()
			self._enhancedFrame = self.mask*self.enhanceFrame(self._currentframeData)
		ROIButton.on_clicked(setROI)
		
		calibButton = Button(plt.axes([0.72,0.87,0.22,0.05]),'Check Calibration',hovercolor='r')
		
		def checkCalibration(event):
			if not self.DISABLE_CCHECK:	
				if self.wireframeon == None:
					self.wireframeimg = frameax.imshow(self.wireframe,alpha=0.5)
					self.wireframeon = True
					return
			
			
				if self.wireframeon:
					#Already displaying so refresh screen
					frameax.clear()
					frame = self.mask*self.enhanceFrame(copy(self._currentframeData))
					self.img = frameax.imshow(frame)
					text = 'Frame: '+str(self._currentframeMark)+'   Time: '+str(self._currentframeTime)+' [ms]'
					self.frametxt = frameax.annotate(text,xy=(0.05,0.95),xycoords='axes fraction',color='white',fontsize=8)
					frameax.set_axis_off()
					frameax.set_xlim(0,frame.shape[1])
					frameax.set_ylim(frame.shape[0],0)
					self.fieldlineArtists = []
					if self.projectLines:
						for line in self.projectLines:
							lplot, = frameax.plot(line[:,0],line[:,1],picker=5,lw=1.5)
							self.fieldlineArtists.append(lplot)
						self.flineTxt = frameax.annotate("Summed intensity: %.2f" % (self.sumIntensity(self.projectLines[-1])),xy=(0.6,0.01),
						xycoords='axes fraction',color='white',fontsize=8)
				else:
					self.wireframeimg = frameax.imshow(self.wireframe,alpha=0.5)
					
				self.wireframeon = not self.wireframeon
			else:
				print("WARNING: Calibration checking disabled!")
				
		calibButton.on_clicked(checkCalibration)
		
		sepButton = Button(plt.axes([0.72,0.93,0.22,0.05]),'Show separatrix',hovercolor='r')
		self.sep_plot = None
		
		def plot_separatrix(event):
			if not self.DISABLE_TRACER:
				if self.sep_plot is None:
					sep_project = self.projectFluxSurface(1.0)	
					self.sep_plot, = frameax.plot(sep_project[:,0],sep_project[:,1],'r--')
					self.sep_plot_on = True
				else:		
					if self.sep_plot_on:
						self.sep_plot.set_ls('')
						self.sep_plot_on = not self.sep_plot_on
					else:
						self.sep_plot.set_ls('--')
						self.sep_plot_on = not self.sep_plot_on

				
		sepButton.on_clicked(plot_separatrix)

		#Some Image analysis options
		correlateButton = Button(plt.axes([0.47,0.23,0.11,0.05]),'Correlate',hovercolor='r')
		
		def getCorrelation(event):
			if not self.selectedPixel:
				return
			corr = self.frameCorrelation((self.selectedPixel[1],self.selectedPixel[0]))
			corrfig,corrax = plt.subplots()
			levels = np.linspace(-1.0,1.0,100)
			corrplot = corrax.imshow(corr,cmap = cm.coolwarm,norm = colors.Normalize(vmin = -1.0, vmax = 1.0, clip = False))
			pixel = corrax.scatter(self.selectedPixel[0],self.selectedPixel[1],s=70,c='g',marker='+')
			corrax.set_title('Correlation of pixel %d,%d' % self.selectedPixel)
			corrax.set_xlim(0,frame.shape[1])
			corrax.set_ylim(frame.shape[0],0)
			cbar = corrfig.colorbar(corrplot)
			plt.show()
		correlateButton.on_clicked(getCorrelation)
			
		timeseriesButton = Button(plt.axes([0.24,0.23,0.22,0.05]),'Intensity Timeseries',hovercolor='r')
		
		def getTimeseries(event):
			if self.selectedLine is not None and len(self.selectedLine) == 2:
				timeseries = self.getIntensityTimeseries(self.selectedLine)
				fig2 = plt.figure()
				plt.subplot(121)
				levels = np.linspace(np.min(timeseries),np.max(timeseries),100)
				plt.contourf(timeseries,levels=levels)
				plt.ylabel('Time index')
				plt.xlabel('Index along line')
				plt.title('Intensity time series')
				delta = self.get_approximate_rad_vel(self.selectedLine[0],self.selectedLine[1],intensity=timeseries,boxcar=10,threshold=2.5,plot=False)
				plt.subplot(122)
				plt.bar([0,1,2,3,4,5,6,7],np.histogram(delta,bins=[0,1,2,3,4,5,6,7,8])[0],edgecolor='red',width=1.0,align='center')
				plt.xlim(0,8)
				plt.title('Widths along line')
				plt.xlabel('Line indices')
				plt.ylabel('counts')
				plt.show()
				
		timeseriesButton.on_clicked(getTimeseries)
				
		distributionButton = Button(plt.axes([0.01,0.23,0.22,0.05]),'Toroidal Distribution',hovercolor = 'r')
		
		def getToroidalDist(event):
			if not self.DISABLE_TRACER:
				intensity = self.toroidalDistribution()
				fig3 = plt.figure()
				phi = (self._currentphi + 360*np.arange(3600)/3599) % 360
				mean = running_mean_periodic(intensity,41)
				phi = phi[np.argsort(phi)]
				plt.plot(phi,intensity,'b.')
				maxima = argrelmax(mean[np.argsort(phi)])[0]
				minima = argrelmin(mean[np.argsort(phi)])[0]
				mean = mean[np.argsort(phi)]
				plt.plot(phi,mean,'r')
				plt.plot(phi[maxima],mean[maxima],'ko')
				plt.plot(phi[minima],mean[minima],'kx')
				plt.ylabel('Summed Intensity (arb)')
				plt.xlabel('Toroidal Angle (degrees)')
				plt.show()
				
			else:
				print("WARNING: Cannot get toroidal intensity distribution, field line tracing disabled!")
		distributionButton.on_clicked(getToroidalDist)
		
		mapButton = Button(plt.axes([0.01,0.16,0.22,0.05]),'Intensity Map',hovercolor = 'r')
		
		def getIntensityMap(event):
			if not self.DISABLE_TRACER:
				self.intensityMap(plot=True)
		mapButton.on_clicked(getIntensityMap)
		
		#Handle Mouse click events
		def onClick(event):
			if event.button == 1:
				if event.dblclick:
					if event.inaxes is frameax:
						self.selectedPixel = (int(event.xdata),int(event.ydata))
						if self.pixelplot:
							self.pixelplot.set_visible(False)
						self.pixelplot = frameax.scatter(self.selectedPixel[0],self.selectedPixel[1],s=70,c='r',marker='+')
						fig.canvas.draw()	
			elif event.button == 3:
				if event.dblclick:
					if event.inaxes is frameax:
						if self.selectedLine is None or len(self.selectedLine) == 2:
							#Load in first pixel coordinate and draw
							if self.linePlot is not None:
								self.linePlot[0].set_visible(False)
								self.linePlot[1].set_visible(False)
								self.linePlot[2][0].remove()
							self.selectedLine = [[int(event.xdata),int(event.ydata)]]
							self.linePlot = [frameax.scatter(int(event.xdata),int(event.ydata),s=70,c='y',marker='x')]
							fig.canvas.draw()
						elif len(self.selectedLine) == 1:
							self.selectedLine.append([int(event.xdata),int(event.ydata)])
							self.linePlot.append(frameax.scatter(int(event.xdata),int(event.ydata),s=70,c='y',marker='x'))
							self.linePlot.append(frameax.plot([self.selectedLine[0][0],self.selectedLine[1][0]],
												[self.selectedLine[0][1],self.selectedLine[1][1]],'-y',lw=2))
							fig.canvas.draw()
												
		fig.canvas.mpl_connect('button_press_event', onClick)	
		
		#Display UI
		plt.show()
	
	
	def enhanceFrame(self,frame):
		#frame = np.uint8(frame*255.0/np.max(frame))			
		if self.applySub:
			frame = self.bgsub.apply(frame)
		frame = cv2.cvtColor(frame,cv2.COLOR_GRAY2BGR)#np.uint8(frame*255.0/np.max(frame)) 
		frame = np.uint8(frame*255.0/np.max(frame))
		if self.noiseRmv:
			try:
				frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
			except:
				pass
			frame = cv2.bilateralFilter(frame,5,75,75)
			try:
				frame = cv2.cvtColor(frame,cv2.COLOR_GRAY2BGR)
			except:
				pass
	
		if self.threshold:
			_,frame = cv2.threshold(frame,10,255,cv2.THRESH_BINARY)
		if self.gammaEnhance:
			gammaframe = np.float64(frame)**(self.gamma)
			frame = np.uint8(gammaframe*255.0/np.max(gammaframe))
		if self.histEq:
			try:
				frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
			except:
				pass
			frame = cv2.equalizeHist(frame)
			try:
				frame = cv2.cvtColor(frame,cv2.COLOR_GRAY2BGR)
			except: 
				pass
		if self.edgeDet:
			try:
				frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
			except:
				pass
			frame = cv2.Canny(frame,500,550,True)
			try:
				frame = cv2.cvtColor(frame,cv2.COLOR_GRAY2BGR)
			except:
				pass

		if self.negative:
			frame = 255 - frame		

		#frame = np.uint8(frame*255.0/np.max(frame))	
		return frame
		
	def projectFieldline(self,fieldline):
		objpoints = np.array([[fieldline.X[i],fieldline.Y[i],fieldline.Z[i]] for i in np.arange(len(fieldline.X))])
		return self.calibration.ProjectPoints(objpoints)[:,0,:]	
		
	def projectFluxSurface(self,psiN,angle=214.9):
		#Project a flux surface onto the camera FOV
		#Note for MAST, tangency angle is 214.9 degrees
		tan_ang = 2.0*np.pi*angle/360.0
		sep = self.tracer.eq.get_fluxsurface(psiN)
		Z = sep.Z
		X = sep.R*np.cos(tan_ang)
		Y = sep.R*np.sin(tan_ang)
		objpoints = np.array([[X[i],Y[i],Z[i]] for i in np.arange(len(X))])
		return self.calibration.ProjectPoints(objpoints)[:,0,:]	
	
	def nextFrame(self):
		self._currentframeNum += 1
		if self._currentframeNum >= self.frames[...].shape[0]:
			self._currentframeNum = self.frames[...].shape[0] - 1
		self._currentframeData = self.frames[self._currentframeNum]
		self._currentframeDisplay = cv2.cvtColor(self._currentframeData,cv2.COLOR_GRAY2BGR)
		self._currentframeTime = self.frames.timestamps[self._currentframeNum]
		self._currentframeMark = self.frames.frameNumbers[self._currentframeNum]
		
	def previousFrame(self):
		self._currentframeNum -= 1
		if self._currentframeNum < 0:
			self._currentframeNum = 0
		#self._currentframe = self.frames[self._currentframeNum]
		self._currentframeData = self.frames[self._currentframeNum]
		self._currentframeDisplay = cv2.cvtColor(self._currentframeData,cv2.COLOR_GRAY2BGR)
		self._currentframeTime = self.frames.timestamps[self._currentframeNum]
		self._currentframeMark = self.frames.frameNumbers[self._currentframeNum]
		
	def sumIntensity(self,line,frame=None):
		total = 0.0
		N = 0.0
		if frame is None:
			frame = self._enhancedFrame
		try:
			frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
		except:
			pass
				
		for i in np.arange(line.shape[0]):
			yind,xind = int(line[i,0]),int(line[i,1])
			if xind > 0 and xind < frame.shape[0] and yind > 0 and yind < frame.shape[1]:
				total += copy(frame[xind,yind])
				N += 1.0
			else:	
				pass
		return total#/N
  
	def getIntensityTimeseries(self,lineCoords):
		#Need to calculate the indices along the line
		x0,y0 = lineCoords[0][0],lineCoords[0][1]
		x1,y1 = lineCoords[1][0],lineCoords[1][1]

		gradient = (float(y1) - float(y0))/(float(x1) - float(x0))
		intercept = y0 - gradient*x0
		dims = self.frames[:].shape
		timeseries = np.zeros((dims[0],abs(x1-x0)))
			
		for t in np.arange(dims[0]):
			i = 0
			currentFrame = cv2.cvtColor(self.enhanceFrame(self.frames[t]),cv2.COLOR_BGR2GRAY)
			#print currentFrame.shape
			for x in np.linspace(x0,x1,abs(x1-x0)):
				y = gradient*x + intercept
				timeseries[t,i] = currentFrame[y,x]
				i += 1

		return timeseries
  
	def frameCorrelation(self,coords=(0,0),delay=0):
		dims = self.frames[:].shape
		frames = np.empty((dims[0]-abs(delay),dims[1],dims[2]))
		
		for i in np.arange(dims[0]-abs(delay)):
			frames[i] = cv2.cvtColor(self.enhanceFrame(self.frames[i]),cv2.COLOR_BGR2GRAY)
			
		#Get pixel means and standard deviations
		frames -= frames.mean(axis=0)
		frames /= frames.std(axis=0)
		
		result = np.zeros((frames.shape[1],frames.shape[2]))
		if delay > 0:
			for x in np.arange(dims[1]):
				for y in np.arange(dims[2]):
					result[x,y] = np.mean(frames[delay:,coords[0],coords[1]]*frames[0:-delay,x,y])
		elif delay < 0:
			for x in np.arange(dims[1]):
				for y in np.arange(dims[2]):
					result[x,y] = np.mean(frames[0:delay,coords[0],coords[1]]*frames[-delay:,x,y])
		else:
			for x in np.arange(dims[1]):
				for y in np.arange(dims[2]):
					result[x,y] = np.mean(frames[:,coords[0],coords[1]]*frames[:,x,y])
					
		return result
		
	def toroidalDistribution(self,nphi=3600,nR=0,Rstart=None,Rend=None,dR=None,verbose=False,frame=None):
		dphi = 2.0*np.pi/nphi
		if nR == 0:
			intensity = []
			if Rstart is None:
				Rstart = self._currentR
			fieldline = self.tracer.trace(Rstart,self._currentZ,phistart = 2.0*np.pi*self._currentphi/360.0,mxstep=1000,ds=0.05)
			for i in np.arange(nphi):
				if verbose:
					print("phi = %.2f" % (self._currentphi + i*dphi))
				linepoints = self.projectFieldline(fieldline)
				intensity.append(self.sumIntensity(linepoints,frame=frame))
				fieldline.rotate_toroidal(dphi)
		else:
			if Rstart is None:
				Rstart = self._currentR
			if dR is None:
				if Rend is None:
					Rend = Rstart + 0.1
				dR = (Rend - Rstart)/nR
			fieldlines = [self.tracer.trace(Rstart + j*dR,self._currentZ,phistart = 0.0,mxstep=1000,ds=0.05) for j in np.arange(nR)]
			intensity = np.zeros([nR,nphi])
			j = 0
			for fieldline in fieldlines:
				if verbose:
					print("R = %.2f" % (Rstart + j*dR))
				for i in np.arange(nphi):
					linepoints = self.projectFieldline(fieldline)
					if verbose:
						print("phi = %.2f" % (float(i)*dphi))
					intensity[j,i] = self.sumIntensity(linepoints,frame=frame)
					fieldline.rotate_toroidal(dphi)
				j += 1
				

		return intensity
	
	def intensityMap(self,plot=False):
		path = self._gfile.split('/')[-1]
		nR = 100
		nphi = 200
		try:
			fieldlines = pickle.load(open('fieldline_store/'+path+'__fl.p','r'))
		except:
			fieldlines = []
			Rstart = 1.37
			phistart = 160.0
			dR = 0.001
			dphi = 0.2
			for i in np.arange(nR):
				R = Rstart + i*dR
				fline = self.tracer.trace(R,0.0,phistart*2.0*np.pi/360.0,mxstep=10000,ds=0.01)
				for j in np.arange(nphi):
					fline.rotate_toroidal(dphi*2.0*np.pi/360.0)
					line = self.projectFieldline(fline)
					indsR = np.where(np.array(fline.R) > 0.6)[0]
					inds = np.where(np.abs(fline.Z)[indsR] < 1.0)[0]
					fieldlines.append(line[inds])
			pickle.dump(fieldlines,open('fieldline_store/'+path+'__fl.p','w'))		
		frame = self.bgsub.apply(self._currentframeData)
		intensity = [] 			
		for i in np.arange(len(fieldlines)):
			line = fieldlines[i]
			temp = []
			points = np.zeros(frame.shape)
			for j in np.arange(line.shape[0]):
				yind,xind = int(line[j,0]),int(line[j,1])
				if xind > 0 and xind < frame.shape[0] and yind > 0 and yind < frame.shape[1]:
					points[xind,yind] = 1.0
			intensity.append(np.sum(points*frame))	
		
		intensity = np.array(intensity).reshape((nR,nphi))
		if plot:
			fig = plt.figure()
			levels = np.linspace(np.min(intensity),np.max(intensity),200)
			plt.contourf(np.linspace(1.37,1.47,100),np.linspace(160,200,200),intensity.T,levels=levels)
			plt.show()
		else:	
			return intensity		
				
	def get_fft(self,subtract=False):
		self.fftfig = plt.figure(figsize=(8,8),facecolor='w')
		fftax1 = plt.axes([0.1,0.2,0.25,0.6])
		fftax2 = plt.axes([0.4,0.2,0.25,0.6])
		fftax3 = plt.axes([0.7,0.2,0.25,0.6])
		frames = np.zeros((self.frames[:].shape[0],self.frames[:].shape[1],self.frames[:].shape[2]))
		for i in np.arange(frames.shape[0]):
			if subtract:   frames[i] = cv2.cvtColor(cv2.cvtColor(self.bgsub.apply(self.frames[i]),cv2.COLOR_GRAY2BGR),cv2.COLOR_BGR2GRAY)
			else:	frames[i] = cv2.cvtColor(cv2.cvtColor(self.frames[i],cv2.COLOR_GRAY2BGR),cv2.COLOR_BGR2GRAY)
		fft = np.fft.rfft(frames,axis=0) 
		rate = 1.0/(self.frames.timestamps[1] - self.frames.timestamps[0])
		self.__power = np.abs(fft[1:])#**2.0
		self.__freqs = np.linspace(0,rate/2,self.__power.shape[0] + 1)[1:]
		#self.__power /= np.sum(self.__power,axis=0)
		self.fftimg1 = fftax1.imshow(self.__power[0],cmap=cm.coolwarm)#'afmhot')
		self.fftimg2 = fftax2.imshow((self.__power/np.sum(self.__power,axis=0))[0],cmap=cm.coolwarm)#'afmhot')
		self.fftimg3 = fftax3.imshow(((self.__power/np.sum(self.__power,axis=0))[0])/np.max((self.__power/np.sum(self.__power,axis=0))[0]),cmap=cm.coolwarm)#'afmhot')
		fftax1.set_axis_off()
		fftax1.set_title('Power')
		fftax2.set_axis_off()
		fftax2.set_title('Normalized')
		fftax3.set_axis_off()
		fftax3.set_title('Relative')
		freq_slider = Slider(plt.axes([0.3,0.08,0.5,0.03]),'Frequency (kHz)',self.__freqs[0]/1000.0,self.__freqs[-1]/1000.0,valinit=self.__freqs[0]/1000.0,valfmt=u'%.1f')

		def on_change(val):
			ind = np.abs(self.__freqs - val*1000.0).argmin()
			self.fftimg1.set_data(self.__power[ind])
			self.fftimg2.set_data((self.__power/np.sum(self.__power,axis=0))[ind])
			self.fftimg3.set_data(((self.__power/np.sum(self.__power,axis=0))[ind])/np.max((self.__power/np.sum(self.__power,axis=0))[ind]))
			self.fftfig.canvas.draw()

		freq_slider.on_changed(on_change)	
		
		plt.show()

	def get_moments(self,subtract=False):
		#momntfig = plt.figure(figsize=(8,8),facecolor='w')
		momntax1 = plt.axes([0.0,0.2,0.25,0.6])
		momntax1.set_axis_off()
		momntax1.set_title('Mean')
		momntax2 = plt.axes([0.25,0.2,0.25,0.6])
		momntax2.set_axis_off()
		momntax2.set_title('Variance')
		momntax3 = plt.axes([0.5,0.2,0.25,0.6])
		momntax3.set_axis_off()
		momntax3.set_title('Skewness')
		momntax4 = plt.axes([0.75,0.2,0.25,0.6])
		momntax4.set_axis_off()
		momntax4.set_title('Kurtosis')
	
		frames = np.zeros((self.frames[:].shape[0],self.frames[:].shape[1],self.frames[:].shape[2]))
		for i in np.arange(frames.shape[0]):
			
			if subtract: frames[i,0:,0:] = cv2.cvtColor(cv2.cvtColor(self.bgsub.apply(self.frames[i]),cv2.COLOR_GRAY2BGR),cv2.COLOR_BGR2GRAY)
			else: frames[i,0:,0:] = cv2.cvtColor(cv2.cvtColor(self.frames[i],cv2.COLOR_GRAY2BGR),cv2.COLOR_BGR2GRAY)		
		frames /= np.max(frames)		
	
		nobs,minmax,mean,var,skew,kurt = describe(frames,axis=0)

		im1 = momntax1.imshow(mean,cmap='afmhot')
		im2 = momntax2.imshow(var,cmap='afmhot')
		im3 = momntax3.imshow(skew,cmap='bwr',vmin=-8.0,vmax=8.0)
		im4 = momntax4.imshow(kurt,cmap='afmhot',vmax=64.0)
		cax1 = plt.axes([0.01,0.1,0.23,0.04])
		cax2 = plt.axes([0.26,0.1,0.23,0.04])
		cax3 = plt.axes([0.51,0.1,0.23,0.04])
		cax4 = plt.axes([0.76,0.1,0.23,0.04])
		plt.colorbar(im1,cax=cax1,orientation='horizontal')
		plt.colorbar(im2,cax=cax2,orientation='horizontal')
		plt.colorbar(im3,cax=cax3,orientation='horizontal')
		plt.colorbar(im4,cax=cax4,orientation='horizontal')
		plt.show()

		return mean,var,skew,kurt

	def get_svd(self,nmom=None,bgsub=True):

		if bgsub: frame = np.float64(self.bgsub.apply(self._currentframeData))
		else: frame = np.float64(self._currentframeData)
		S,u,v = cv2.SVDecomp(frame)

		size = frame.shape
		print S.shape
		print u.shape
		print v.shape
		
		nx = size[0]
		ny = size[1]
		reconst = np.zeros(size)
		if nmom is None:
			nmom = nx
		print reconst.shape
		for i in np.arange(nx):
			for j in np.arange(nmom-1):
				reconst[i,:] = reconst[i,:] +  u[i,j]*S[j,0]*v[j,:]

		print reconst.shape

		plt.imshow(np.concatenate([frame,reconst],axis=1),cmap='gray')
		plt.title(str(nmom)+" Moments")
		plt.show()
			
	def fft_denoise(self,x=True,frame=None,nmom = 5):
		if x: axis = 0
		else: axis = 1 
		if frame is None: frame = self.bgsub.apply(self._currentframeData)
		fft = np.fft.rfft(frame,axis=axis)
		if axis == 1: plt.imshow(np.log(np.abs(fft[:,1:])**2.0))
		else: plt.imshow(np.log(np.abs(fft[1:])**2.0))
		plt.show()
		if nmom > 0:	
			if axis == 0: fft[-nmom:,:] = 0.0
			else: fft[:,-nmom:] = 0.0
		else:
			if axis == 0: fft[0:-nmom,:] = 0.0
			else: fft[:,0:-nmom] = 0.0
		ret_frame = np.fft.irfft(fft,axis=axis)	
		plt.imshow(np.concatenate([frame,ret_frame],axis=1),cmap='gray')
		plt.show()

	def get_fieldline_convolution(self,R,Z,phi):
		fieldline = self.tracer.trace(R,Z,phi*2.0*np.pi/360.0,mxstep=10000,ds=0.005)
		line = self.projectFieldline(fieldline)
		indsR = np.where(np.array(fieldline.R) > 0.6)[0]
		inds = np.where(np.abs(fieldline.Z)[indsR] < 1.0)[0]
		line = line[inds]
		frame = self._currentframeData
		frame_sub = self.bgsub.apply(self._currentframeData)
		point_array = np.zeros(frame.shape)
		for j in np.arange(line.shape[0]):
			yind,xind = int(line[j,0]),int(line[j,1])
			if xind > 0 and xind < frame.shape[0] and yind > 0 and yind < frame.shape[1]:
				point_array[xind,yind] = 1.0
		plt.imshow(point_array,alpha=0.4)
		#plt.imshow(frame_sub)
		plt.show()
		from scipy.signal import convolve2d
		conv = convolve2d(frame_sub,point_array)
		plt.imshow(conv)
		plt.show()
		
	def get_approximate_rad_vel(self,pix1,pix2,intensity = None,startframe=0,endframe=-1,boxcar=7,threshold=2.5,plot=True,velocity=True):
		if intensity is None: intensity = self.getIntensityTimeseries([pix1,pix2])[startframe:endframe]
		nt = intensity.shape[0]
		nr = intensity.shape[1]	
		delta = []
		vr = []
		if plot: plt.figure()
		time_series = running_mean(intensity[:,int(nr/4)],boxcar)
		maxima = argrelmax(time_series)
		minima = argrelmin(time_series)
		if plot:
			plt.plot(time_series)		
			plt.plot(maxima[0],time_series[maxima[0]],'ro')
			plt.plot(minima[0],time_series[minima[0]],'rx')
			plt.title('Detected filaments')
			plt.xlabel('time index')
			plt.ylabel('Intensity (arb)')
			plt.show()
			plt.figure()		
		for value in maxima[0]:
			rad_series = running_mean(intensity[value],3)
			mean = np.mean(rad_series)
			rad_maxima = argrelmax(rad_series)[0]
			rad_minima = argrelmin(rad_series)[0]
			#plt.plot(rad_maxima,rad_series[rad_maxima],'ko')
			#plt.plot(rad_minima,rad_series[rad_minima],'kx')
			for value in rad_maxima:
				if rad_series[value] > threshold*mean:
					#Accept as a filament
					imin = value
					while rad_series[imin] > threshold*mean:
						imin -= 1
						if imin == 0:
							break
					
					imax = value
					while rad_series[imax] > threshold*mean:
						imax += 1
						if imax == nr-1:
							break
					if imin > 0 and imax < nr-1: 
						delta.append(float((imax-imin)))
						if plot: plt.plot(rad_series)
						
		if plot:
			plt.title('Detected filaments')
			plt.xlabel('Line index')
			plt.show()
			plt.figure()
			plt.bar([0,1,2,3,4,5,6,7],np.histogram(delta,bins=[0,1,2,3,4,5,6,7,8])[0],edgecolor='red',width=1.0,align='center')
			plt.xlim(0,8)
			plt.title('Widths along line')
			plt.xlabel('Line indices')
			plt.ylabel('counts')
			plt.show()
	
		return delta


	def get_autocorrelation(self,subtract=False,nt=100):
		frames = np.zeros((self.frames[:].shape[0],self.frames[:].shape[1],self.frames[:].shape[2]))
		for i in np.arange(frames.shape[0]):
			if subtract: frames[i] = self.bgsub.apply(self.frames[i])
			else: frames[i] = self.frames[i]
		frames -= frames.mean(axis=0)
		frames /= frames.std(axis=0)
		corr = np.zeros((nt,frames.shape[1],frames.shape[2]))
		for delay in np.arange(nt):
			corr[delay] = (frames*np.roll(frames,-delay,axis=0)).mean(axis=0)
		auto_corr = np.zeros((frames.shape[1],frames.shape[2]))
		#auto_corr = 1.0 + 2.0*np.sum(corr,axis=0)
		#auto_corr *= (self.frames.timestamps[1]-self.frames.timestamps[0])
		for x in np.arange(frames.shape[1]):
			for y in np.arange(frames.shape[2]):
				for i in np.arange(nt):
					if corr[i,x,y] < 1.0/2.718:
						auto_corr[x,y] = i
						break
		auto_corr *= (self.frames.timestamps[1]-self.frames.timestamps[0])
		return auto_corr			
if __name__=='__main__':
	import sys
        try:
		shot = sys.argv[1]
	except:
		shot = None
		pass
		
	#Elzar = Elzar(Nframes=100,startpos=0.5,shot=shot,time=0.272)
	moviefile = '/Users/Nick/MAST_MOVIES/rbf029852.ipx'
	moviefile = '/Volumes/NO NAME/PHOTRON/50421/50421.mraw'
	Elzar = Elzar(Nframes=50,startpos=0.0,moviefile=moviefile)#,time=0.3,gfile='../../gfiles/g_p29852_t0.216',calibfile='29852')
	#Elzar.get_svd(10)	
	#Elzar.get_fft()
	#Elzar.get_moments()
	#Elzar.intensityMap(plot=True)
	Elzar.runUI()	
	#mapp = Elzar.intensityMap(plot=True)
	#import pickle	
	#plt.imshow(Elzar.bgsub.backgroundModel)
	#plt.show()
	#print Elzar.bgsub.backgroundModel.shape
	#print Elzar._currentframeData.shape
	#print np.max(Elzar.bgsub.backgroundModel)
	#print np.max(Elzar._currentframeData)
	#plt.imsave('bgmodel_21695',Elzar.bgsub.backgroundModel)
	#pickle.dump(Elzar.bgsub.backgroundModel,open('bgmodel_21695.p','wb'))

	#frame = pickle.load(open('29852_sim_movie_0.p','rb'))
	#Elzar.frames[0] = frame

	#Elzar.run_fluxtube_launcher()
	#intensity = Elzar.toroidalDistribution(Rstart = 1.47,nR=20,Rend=1.60,nphi=1000,verbose=True,frame=Elzar.mask*Elzar.enhanceFrame(Elzar.frames[10]))
	#levels = np.linspace(np.min(intensity),np.max(intensity),100)
	#plt.contourf(np.linspace(0.0,360.0,1000),np.linspace(1.47,1.60,20),intensity,levels=levels)
	#plt.colorbar()	
	##plt.xlabel('Toroidal Angle (deg)')
	#plt.ylabel('Major radius (m)')
	#plt.show()


	
