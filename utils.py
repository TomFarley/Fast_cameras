#!/usr/bin/env python

"""
Utility functions used in the filament tracker program
"""

from Tkinter import Tk
from tkFileDialog import askopenfilename
from matplotlib.widgets import RectangleSelector
import numpy as np

def getUserFile(type=""):	
	Tk().withdraw()
	filename = askopenfilename(message="Please select "+type+" file:")	
	return filename


def findNearest(inp,(valx,valy)):
    inparr = np.asarray(inp)
    return (((inparr[:,0]-valx)**2.0 + (inparr[:,1] - valy)**2.0)**0.5).argmin()
        
	
from matplotlib import cbook
class DataCursor(object):
    """A simple data cursor widget that displays the x,y location of a
    matplotlib artist when it is selected."""
    def __init__(self, artists,func=None, tolerance=5, offsets=(-20, 20), 
                 template='x: %0.2f\ny: %0.2f', display_all=False):
        """Create the data cursor and connect it to the relevant figure.
        "artists" is the matplotlib artist or sequence of artists that will be 
            selected. 
        "tolerance" is the radius (in points) that the mouse click must be
            within to select the artist.
        "offsets" is a tuple of (x,y) offsets in points from the selected
            point to the displayed annotation box
        "template" is the format string to be used. Note: For compatibility
            with older versions of python, this uses the old-style (%) 
            formatting specification.
        "display_all" controls whether more than one annotation box will
            be shown if there are multiple axes.  Only one will be shown
            per-axis, regardless. 
        """
        self.template = template
        self.offsets = offsets
        self.display_all = display_all
        self.func = func
        if not cbook.iterable(artists):
            artists = [artists]
        self.artists = artists
        self.axes = tuple(set(art.axes for art in self.artists))
        self.figures = tuple(set(ax.figure for ax in self.axes))
        self.ids = []
        self.annotations = {}
        for ax in self.axes:
            self.annotations[ax] = self.annotate(ax)

        for artist in self.artists:
            artist.set_picker(tolerance)
        

    def annotate(self, ax):
        """Draws and hides the annotation box for the given axis "ax"."""
        annotation = ax.annotate(self.template, xy=(0, 0), ha='left',
                xytext=self.offsets, textcoords='offset points', va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0')
                )
        annotation.set_visible(False)
        return annotation

    def __call__(self, event):
        """Intended to be called through "mpl_connect"."""
        # Rather than trying to interpolate, just display the clicked coords
        # This will only be called if it's within "tolerance", anyway.
        x, y = event.mouseevent.xdata, event.mouseevent.ydata
        annotation = self.annotations[event.artist.axes]
        if x is not None:
            if not self.display_all:
                # Hide any other annotation boxes...
                for ann in self.annotations.values():
                    ann.set_visible(False)
            # Update the annotation in the current axis..
            annotation.xy = x, y
            if self.func is None:
               
                annotation.set_text(self.template % (x, y))        
                annotation.set_visible(True)
                event.canvas.draw()
            else:
                self.func(event,annotation)
  
    def connect(self,fig):
        self.cid = fig.canvas.mpl_connect('pick_event', self) 
                  
    def disconnect(self,fig):
        fig.canvas.mpl_disconnect(self.cid)
	     
    def clear(self,fig):
        for ann in self.annotations.values():
            ann.set_visible(False)
        fig.canvas.draw()
        
        
class ROISelector(object):
    
    def __init__(self,artist):
            self.artist = artist
            self.selector = RectangleSelector(self.artist.axes,self.on_select,
                                       button=3, minspanx=5, minspany=5, spancoords='pixels',
                                       rectprops = dict(facecolor='red', edgecolor = 'red',
                                                        alpha=0.3, fill=True))
            self.coords = []
            
    def on_select(self,click,release):
            x1,y1 = int(click.xdata),int(click.ydata)
            x2,y2 = int(release.xdata),int(release.ydata)
            self.coords =[(x1,y1),(x2,y2)]
            
    def activate(self):
        self.selector.set_active(True)
        
    def deactivate(self):
        self.selector.set_active(False)        
    

def running_mean_periodic(series,window):
	""" Compute the running mean of a 1D sequence """
	input = np.asarray(series)
	output = []
	if window % 2 == 0:
		width = window/2
	else:
		width = (window - 1)/2

	for i in np.arange(input.shape[0]):
		if i - width < 0:
			temp = np.concatenate((input[i-width:],input[0:i+width]))
		elif i + width > input.shape[0]:
			temp = np.concatenate((input[i-width:-1],input[0:i + width - input.shape[0]]))
		else:
			temp = input[i-width:i+width]

		output.append(np.mean(temp))

	return np.asarray(output)
       
def running_mean(series,window):
        """ Compute the running mean of a 1D sequence """
        input = np.asarray(series)
        output = []
        if window % 2 == 0:
                width = window/2
        else:
                width = (window - 1)/2 
        for i in np.arange(input.shape[0]-(2*width))+width:
                if i - width < 0:
                        temp = np.concatenate((input[i-width:],input[0:i+width]))
                elif i + width > input.shape[0]:
                        temp = np.concatenate((input[i-width:-1],input[0:i + width - input.shape[0]]))
                else:
                        temp = input[i-width:i+width]

                output.append(np.mean(temp))
	result = np.concatenate((input[0:width],np.asarray(output),input[-width:]))


        return result
                  
   
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    plt.figure()
    plt.subplot(2,1,1)
    line1, = plt.plot(range(10), 'ro-')
    plt.subplot(2,1,2)
    line2, = plt.plot(range(10), 'bo-')

    DataCursor([line1, line2])

    plt.show()
