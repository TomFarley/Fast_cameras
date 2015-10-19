import cv2
import numpy as np
import struct

"""
Module containing a reader for .ipx files.

To use this to play a movie simply run the module with the shot number as a command line argument, i.e

python ipxReader.py 28819 

will play the .ipx movie in shot 28819
"""

class movieReader(object):
	"""
	Base class for movie readers
	"""

	def __init__(self,filename=None):		

		self.file_header = {}
		self._current_position =  0
		self._current_frame = 0 

		if filename is not None:
			self.open(filename)


	def open(self,file):
		#try:
		self._open(file)
		#except:
		#raise IOError("ERROR: Failed to open file : "+file)

	def read(self):
		return self._read()

	def set_frame_number(self,frame_number):
		self._set_frame_number(frame_number)
	
	def set_frame_time(self,frame_time):
		self._set_frame_time(frame_time)

	def reset(self):
		self._reset()

	def release(self):
		self._file.close()

	def get(self,field):
		try:
			return self.file_header[field]
		except:	
			print("Property "+field+" not found in header of file "+self._filename+"\n")
			return None

class ipxReader(movieReader):
	"""
	Class to read .ipx video files 
	
	Class Attributes:
	
		ipxReader.file_header				Dictionary containing information from the 
											file header
			
	Class Methods:
	
		ipxReader.open(filename)			Open the ipx file for reading and read in file header
		
		bool,np.ndarray,dict = ipxReader.read()	
								Read the next movie frame, return the frame data as a
								numpy ndarray, return the frame header as a dict
		
		ipxReader.release()				Release the loaded file
		
		ipxReader.get(property)				Return the value of the property if found in the file header

		ipxReader.reset()				Reset the reader to begin at the start of the file again

		ipxReader.set_frame_number(number)		Set the reader to read at the frame number given

		ipxReader.set_time(time)			Set the reader to read at the time given

	Instantiate using 

		myReader = ipxReader()				Returns a bare reader that is not linked to any file
		myReader = ipxReader(shot=xxxxx)		Returns a reader linked to the rbb0xxxxx.ipx file in the $MAST_IMAGES
								directory
		mtReader = ipxReader(filename='myfile.ipx')	Returns a reader linked to the file 'myfile.ipx'	
					
				NOTE: the shot keyword takes precedence over the filename keyword so 

					myReader = ipxReader(filename='myfile.ipx',shot=99999) 
			
				will be linked to shot 99999, rather than myfile.ipx
				
	"""
	
	
	#Store .ipx formatting lists as class variables 
	__IPX_HEADER_FIELDS = ['ID','size','codec','date_time','shot',
							'trigger','lens','filter','view','numFrames',
							'camera','width','height','depth','orient',
							'taps','color','hBin','left','right','vBin',
							'top','bottom','offset_0','offset_1','gain_0',
							'gain_1','preExp','exposure','strobe','board_temp',
							'ccd_temp']
							
	__IPX_HEADER_LENGTHS = [8,4,8,20,4,4,24,24,64,4,64,2,2,2,4,
							2,2,2,2,2,2,2,2,2,2,4,4,4,4,4,4,4]
							
	__IPX_HEADER_OFFSETS = [0]	
	for length in __IPX_HEADER_LENGTHS:
		__IPX_HEADER_OFFSETS.append(__IPX_HEADER_OFFSETS[-1]+length)
	__IPX_HEADER_OFFSETS.pop()
		
	# s = string					
	# I = unsigned int
	# i = int
	# f = float
	# H = unsigned short					
	__IPX_HEADER_TYPES = ['s','I','s','s','i','f','s','s','s',
						'I','s','H','H','H','I','H','H','H',
						'H','H','H','H','H','H','H','f','f',
						'I','I','I','f','f']
				
	def __init__(self,filename=None,shot=None,camera='rbb'):
		
		movieReader.__init__(self,filename=filename)		
		if shot:
			shot_string = str(shot)
			shot_string_list = list(shot_string)
			MAST_PATH = "/net/fuslsa/data/MAST_IMAGES/0"+shot_string_list[0]+shot_string_list[1]+"/"+shot_string+"/"+camera+"0"+shot_string+".ipx"
			self.open(MAST_PATH)

				
	def _open(self,filename):

		self._file = open(filename,'rb')
		self._filename = filename
			
		#Read and store the file header	
		for i in np.arange(len(self.__IPX_HEADER_FIELDS)):
			self._file.seek(self.__IPX_HEADER_OFFSETS[i])
			
			try:
				if self.__IPX_HEADER_TYPES[i] == 's':
					format_string = '<'+str(self.__IPX_HEADER_LENGTHS[i])+self.__IPX_HEADER_TYPES[i]	
				else:
					format_string = '<'+self.__IPX_HEADER_TYPES[i]	
				self.file_header[self.__IPX_HEADER_FIELDS[i]], = struct.unpack(format_string,self._file.read(self.__IPX_HEADER_LENGTHS[i]))
			except:
				print("WARNING: Unable to read Header field "+self.__IPX_HEADER_FIELDS[i])
				self.file_header[self.__IPX_HEADER_FIELDS[i]] = None
					
		self._current_position = self.file_header['size']
	
	def _read_frame_header(self):
		try:
			if self._file is None:
				print("WARNING: No file opened, returning")
				return
			
			self._file.seek(self._current_position)
			
			#Read the frame header
			header = {}
			header['size'], = struct.unpack('<I',self._file.read(4))
			self._current_position += 4
			
			self._file.seek(self._current_position)
			header['time_stamp'], = struct.unpack('<d',self._file.read(8))
			self._current_position += 8
			
			return True,header
	    	
		except:
			#Ensure that the reader closes gracefully
			print("\nWARNING: End of file detected. Closeing.")			
			self.release()
			
			return False,None
		
	def _read(self):		
		try:
			if self._file is None:
				print("WARNING: No file opened, returning")
				return

			#Read the frame header
			ret,header = self._read_frame_header()
			
			#Now read in the frame data as a byte array
			self._file.seek(self._current_position)
			format_string = '<'+str(header['size'])+'s'
			byte_data = bytearray(self._file.read(header['size']))
			self._current_position += header['size'] - 12
			
			#Now decode the jpg2 data using opencv, keeping output colorstyle as input colorstyle
			image_data = cv2.imdecode(np.asarray(byte_data,dtype=np.uint8),-1)
			image_data.reshape(self.file_header['height'],self.file_header['width'])
			
			self._current_frame += 1
			
			return True,image_data,header
	    	
		except:
			#Ensure that the reader closes gracefully
			print("\nWARNING: End of file detected. Closeing.")			
			self.release()
			
			return False,None,None

	def _skip_frame(self):
		ret,header = self._read_frame_header()
		self._current_position += header['size'] - 12
		return header

	def _reset(self):
		self._current_position = self.file_header['size']
		self._current_frame = 0	
	
	def _set_frame_number(self,frame_number):
		#always return to the top
		self.reset()
		while self._current_frame < frame_number:
			header = self._skip_frame()
			self._current_frame += 1

	def _set_frame_time(self,set_time):
		self.reset()
		time = 0.0
		while time < set_time:
			header = self._skip_frame()
			self._current_frame += 1
			time = header['time_stamp']


class mrawReader(movieReader):

	def __init__(self,filename=None,shot=None,camera='rbb'):
		movieReader.__init__(self,filename=filename)		
		

	def _open(self,filename):

		filetitle = filename.split('.')[0]
		print filetitle
		#Contains the frames
		try:
			file_mraw = filetitle+".mraw"
			self._file = open(file_mraw,'rb')
		except:
			raise IOError("ERROR: Cannot identify mraw file : "+filetitle+".mraw")

		#Contains the header
		try:
			file_cih = filetitle+".cih"
			cih_file = open(file_cih,'rb')
		except:
			raise IOError("ERROR: Cannot identify header file : "+filetitle+".cih")

		self._filename = filename
		
		#Read and store the file header	
		for line in cih_file:
			print line
			if line[0] == '#':
				pass
			else:
				line_split = line.split(':')
				if len(line_split)>1:
					self.file_header[line_split[0].replace(' ','')] = line_split[1]	
				else:
					pass
		#Size of image frame in bits (no of pixels times size of unsigned int)	
		self._frame_size = int(self.file_header['ImageWidth'])*int(self.file_header['ImageHeight'])*2			
		self._pixels = int(self.file_header['ImageWidth'])*int(self.file_header['ImageHeight'])

	def _read(self):		
		try:
			if self._file is None:
				print("WARNING: No file opened, returning")
				return
			
			#Now read in the frame data as a byte array
			self._file.seek(self._current_position)
			image_data = np.fromfile(self._file,count=self._pixels,dtype=np.uint16)
			image_data = np.reshape(image_data,(int(self.file_header['ImageHeight']),int(self.file_header['ImageWidth'])))
			self._current_frame += 1
			self._current_position += self._frame_size
			try:
				time = float(self.file_header['TriggerTime']) + float(self._current_frame)/float(self.file_header['RecordRate(fps)'])
			except:
				time = float(self.file_header['TriggerTime']) + float(self._current_frame)
			return True,image_data,{'size':self._frame_size,'time_stamp':time}
		except:
			#Ensure that the reader closes gracefully
			print("\nWARNING: End of file detected. Closeing.")			
			self.release()
			
			return False,None,None

	def _skip_frame(self):
		self._current_position += self._frame_size 

	def _reset(self):
		self._current_position = 0
		self._current_frame = 0	
	
	def _set_frame_number(self,frame_number):
		#always return to the top
		self.reset()
		while self._current_frame < frame_number:
			self._skip_frame()
			self._current_frame += 1

	def _set_frame_time(self,set_time):
		self.reset()
		time = 0.0
		while time < set_time:
			self._skip_frame()
			self._current_frame += 1
			time = float(self.file_header['TriggerTime']) + float(self._current_frame)/float(self.file_header['RecordRate(fps)'])


		
if __name__=='__main__':

	import sys
	try:
		shot = sys.argv[1]
	except:
		shot = None

	try:
		cam = str(sys.argv[2])
	except:
		cam = 'rbb'

	A = ipxReader(shot=shot,camera=cam)
	#A = ipxReader(filename="/net/edge1/scratch/jrh/SA1/rbf029827.ipx")
        #A = ipxReader(filename="/net/edge1/scratch/jrh/divertor/29564_hmode/output.ipx")
	#A = ipxReader(filename='/home/jrh/jrh_useful_codes/sa1_ak/29771/29771_align.ipx')
	#for key in A.file_header.keys():
	#	print(key + "\t\t:\t"+str(A.file_header[key]))
	
	while True:		
		ret,frame,header = A.read()
		if not ret:
			break
		
		cv2.imshow('Video',frame)
		
		#Display at 25fps (40ms = 1/25)
		k = cv2.waitKey(40) & 0xff
		
		#Exit if esc key is pushed 
		if k == 27:
			break
	A.release()
	
