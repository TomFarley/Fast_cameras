#!/usr/bin/env python
from __future__ import division


"""
Class to read and store useful information concerning an equilibrium in a manner
that abstracts the user from the data source.

Nick Walkden, James Harrison, Anthony field June 2015
"""

import numpy as np
from copy import deepcopy as copy
from collections import namedtuple
fluxSurface = namedtuple('fluxSurface','R Z')
try:
	#Try to import Afields Point class
	from utilities import Point
except:
	#Otherwise just use a named tuple
	Point = namedtuple('Point', 'r z')

from scipy.interpolate import interp1d
from scipy.interpolate import RectBivariateSpline 

class MDS_signal(object):
	"""
	Simple wrapper around mds to get signals  
	"""
	def __init__(self,connection,signal):
		self.data,self.dimensions = connection.get('[{0},dim_of({0})]'.format(signal))

def interp2d(R,Z,field):
   	"""
   	Same functionality as the scipy function of the same name, 
   	but modded to work with RectBivariateSpline
   	"""
	return RectBivariateSpline(R,Z,np.transpose(field))

def save_pyeq(equilibrium,filename):
	try:
		import cpickle as pickle
	except:
		import pickle
	"""
	Simple wrapper around pickle to save equilbria as .pyeq files 
	"""
	if filename.split('.')[-1] is not 'pyeq': filename += '.pyeq'
	file = open(filename,'wb')
	try:
		pickle.dump(equilbrium,file)
	except:
		raise IOError('Error: unable to pickle data to file --> '+filename)
		    
def load_pyeq(equilibrium,filename):
	try:
		import cpickle as pickle		
	except:
		import pickle
		
	#First try loading filename as given
	
	try:
		file = open(filename,'rb')
	except:
		#Next try append the .pyeq filename
		filename += '.pyeq'
		try:
			file = open(filename,'rb')
		except:
			raise IOError('Error: unable to open file --> '+filename)
	
	try:
		eq = pickle.load(file)
	except:
		raise IOError('Error: unable to unpickle data from file --> '+filename)
	
	return eq


class equilibriumField(object):
	"""
	Container for fields of the equilibrium. 

	An equilibrium field can be accessed either by indexing, ie
	
		data = myfield[i,j]

	or as a function, ie

		data = myfield(R,Z)

	NOTE:
		At present, operations on two equilibriumField objects, i.e

		equilibriumField + equilibriumField 

		return the value of their data only as a numpy array, and do not return
		an equilibriumField object
	"""
	def __init__(self,data,function):

		self._data = data			
		self._func = function

	def __getitem__(self,inds):
		return self._data[inds]
	
	def __len__(self):
		return len(self._data)	
	
	def __add__(self,other):
		if type(other) is type(self):
			return self._data + other._data
		else:
			return self._data + other

	def __radd__(self,other):
		return self.__add__(other)

	def __sub__(self,other):
		if type(other) is type(self):
			return self._data - other._data
		else:
			return self._data - other

	def __truediv__(self,other):
		if type(other) is type(self):
			return self._data/other._data
		else:
			return self._data/other
	def __itruediv__(self,other):
		if type(other) is type(self):
			return other._data/self._data
		else:
			return other/self._data

	def __rsub__(self,other):
		return -1.0*(self.__sub__(other))

	def __mul__(self,other):
		if type(other) is type(self):
			return self._data*other._data
		else:
			return self._data*other
	
	def __rmul__(self,other):
		if type(other) is type(self):
			return other._data*self._data
		else:
			return other*self._data
	def __pow__(self,power):
		return self._data**power

	def __setitem__(self,inds,values):
		self._data[inds] = values

	def __call__(self,*args,**kwargs):
		if len(self._data.shape)>1:
			return np.transpose(self._func(*args,**kwargs))
		else:
			return self._func(*args,**kwargs)

class equilibrium(object):
	""" Equilibrium object to be passed to 
	other codes. Holds useful information about the equilbrium in 
	a general way

	Abstracts user from the data source

	Currently the object contains
		R		1d array 	Major radius grid points
		Z		1d array	Vertical height grid points
		psi		2d field	poloidal flux as a function of (R,Z)
		psiN		2d field	produce normalized psi for a given R,Z 
		dpsidR		2d field	dpsi/dR
		dpsidZ		2d field	dpsi/dZ
		BR		2d field	Radial magnetic field
		BZ		2d field	Vertical magnetic field
		Bp		2d field	poloidal magnetic field
		Bt		2d field	toroidal magnetic field
		B		2d field	total magnetic field	
		fpol		1d field	Toroidal flux function, f = R*Bt, as a function of psi
		fpolRZ		2d field	Toroidal flux function in RZ space
		psi_bnd 	float		value of the separatrix poloidal flux
		psi_axis 	float		value of the poloidal flux on axis
		sigBp		int		Sign of the plasma current (determines sign of poloidal field)
		nr		int		Number of grid points in R
		nz		int		number of grid points in Z
		Btcent		float		Vacuum toroidal field at magnetic axis
		Rcent		float		Major radius at magnetic axis
		wall		dict		Dictionary containing the R and Z coordinates of the wall
		nxpt		int		Number of Xpoints
		xpoint		list		list of Xpoint locations as Point types (see above)
		axis		Point		position of the magnetic axis as a Point type (see above)
	"""
	
	def __init__(self,device=None,shot=None,time=None,gfile=None,with_bfield=True,verbose=False,mdsport=None):
		self.psi = None
		self._loaded = False
		self._time = None		
		self.R = None
		self.Z = None
		self.dpsidR = None
		self.dpsidZ = None
		self.BR = None
		self.BZ = None
		self.Bp = None
		self.Bt = None
		self.B = None
		self.fpol = None
		self.fpolRZ = None
		self.psi_bnd = None
		self.psi_axis = None
		self.sigBp = None
		self.nr = 0
		self.nz = 0
		self.Btcent = 0.0
		self.Rcent = 0.0
		self.wall = {}	
		self.nxpt = 0.0
		self.xpoint = []
		self.axis = None			
		self._shot = None
		self.machine = None
		if shot is not None and time is not None:
			if device is 'MAST':
				self.load_MAST(shot,time,with_bfield=with_bfield,verbose=verbose)
			elif device is 'JET':
				self.load_JET(shot,time,with_bfield=with_bfield,verbose=verbose)
			elif device is 'TCV':
				self.load_TCV(shot,time,port=mdsport,with_bfield=with_bfield,verbose=verbose)
			elif gfile is not None:
				self.load_geqdsk(gfile,with_bfield=with_bfield,verbose=verbose)
			else: 
				return	
		else:
			if gfile is not None:
				self.load_geqdsk(gfile,with_bfield=with_bfield,verbose=verbose)
			else:
				return

	def load_geqdsk(self,gfile,with_bfield=True,verbose=False):
		"""
		load equilibrium data from an efit geqdsk file

		arguments:
			gfile = file or str	file to load data from
						can be either a file type object
						or a string containing the file name
		"""			
		try:
			from geqdsk import Geqdsk
		except:
			ImportError("No geqdsk module found. Cannot load from gfile. Returning.")
			return
		
		ingf = Geqdsk(gfile) 
		self.machine = None

		if verbose: print("\n Loading equilibrium from gfile "+str(gfile)+"\n")
		self.nr = ingf['nw']
		self.nz = ingf['nh']
		
		self.R = np.arange(ingf['nw'])*ingf['rdim']/float(ingf['nw'] - 1) + ingf['rleft'] 
		self.Z = np.arange(ingf['nh'])*ingf['zdim']/float(ingf['nh'] - 1) + ingf['zmid'] - 0.5*ingf['zdim'] 


		self._Rinterp,self._Zinterp = np.meshgrid(self.R,self.Z)
		psi = ingf['psirz']
		psi_func = interp2d(self.R,self.Z,-psi)
		self.psi = equilibriumField(-psi,psi_func)

		self.psi_axis = -ingf['simag']
		self.psi_bnd = -ingf['sibry']
		self.sigBp = -(abs(ingf['current'])/ingf['current'])

		
		fpol = ingf['fpol']
		psigrid = np.linspace(self.psi_axis,self.psi_bnd,len(fpol))
		fpol_func = np.vectorize(interp1d,psigrid)
		self.fpol = equilibriumField(fpol,fpol_func)
		
		self._loaded = True
		self.Btcent = ingf['bcentr']
		self.Rcent = ingf['rcentr']
		
		
		psiN_func = interp2d(self.R,self.Z,(self.psi[:] - self.psi_axis)/(self.psi_bnd-self.psi_axis))
		self.psiN = equilibriumField((self.psi[:] - self.psi_axis)/(self.psi_bnd-self.psi_axis),psiN_func)
	
		R = ingf['rlim']
		Z = ingf['zlim']
		
		self.wall = { 'R' : R, 'Z' : Z }
		self.calc_bfield()

	def load_MAST(self,shot,time,with_bfield=True,verbose=False):
		""" Read in data from MAST IDAM

		arguments:
			shot = int	shot number to read in
			time = float	time to read in data at
		"""
			
		try:
			import idam
		except:
			raise ImportError("No Idam module found, cannot load MAST shot")

		if verbose: print("\nLoading equilibrium from MAST shot "+str(shot)+"\n")

		self.machine = 'MAST'
		self._shot = shot
		if not self._loaded:
			self._psi = idam.Data("efm_psi(r,z)",shot,"fuslwn")
			self._r   = idam.Data("efm_grid(r)",shot,"fuslwn")
			self._z   = idam.Data("efm_grid(z)",shot,"fuslwn")
			self._psi_axis = idam.Data("efm_psi_axis",shot,"fuslwn")
			self._psi_bnd  = idam.Data("efm_psi_boundary",shot,"fuslwn") 
			self._cpasma = idam.Data("efm_plasma_curr(C)",shot,"fuslwn")
			self._bphi = idam.Data("efm_bvac_val",shot,"fuslwn")
			self._xpoint1r = idam.Data("efm_xpoint1_r(c)",shot,"fuslwn")
			self._xpoint1z = idam.Data("efm_xpoint1_z(c)",shot,"fuslwn")
			self._xpoint2r = idam.Data("efm_xpoint2_r(c)",shot,"fuslwn")	
			self._xpoint2z = idam.Data("efm_xpoint2_z(c)",shot,"fuslwn")	
			self._axisr = idam.Data("efm_magnetic_axis_r",shot,"fuslwn")
			self._axisz = idam.Data("efm_magnetic_axis_z",shot,"fuslwn")

		tind = np.abs(self._psi.time - time).argmin()
		self.R = self._r.data[0,:]
		self.Z = self._z.data[0,:]

		psi_func = interp2d(self.R,self.Z,-self._psi.data[tind,:,:])
		self.psi = equilibriumField(-self._psi.data[tind,:,:],psi_func) 

		self.nr = len(self.R)
		self.nz = len(self.Z)
		
		tind_ax = np.abs(self._psi_axis.time - time).argmin()
		self.psi_axis = -self._psi_axis.data[tind_ax]
		tind_bnd = np.abs(self._psi_bnd.time - time).argmin()
		self.psi_bnd = -self._psi_bnd.data[tind_bnd] 
		self.Rcent = 1.0 # Hard-coded(!)
		tind_Bt = np.abs(self._bphi.time - time).argmin()
		self.Btcent = self._bphi.data[tind_Bt]
		tind_sigBp = np.abs(self._cpasma.time - time).argmin()
		self.sigBp = -(abs(self._cpasma.data[tind_sigBp])/self._cpasma.data[tind_sigBp])
	
		self.nxpt = 2
		tind_xpt = np.abs(self._xpoint1r.time - time).argmin()
		self.xpoint = [Point(self._xpoint1r.data[tind_xpt],self._xpoint1z.data[tind_xpt])]
		self.xpoint.append(Point(self._xpoint2r.data[tind_xpt],self._xpoint2z.data[tind_xpt]))
		self.axis = Point(self._axisr.data[tind_xpt],self._axisz.data[tind_xpt])
		
		self.fpol = None		

		self._loaded = True
		self._time = self._psi.time[tind]

		psiN_func = interp2d(self.R,self.Z,(self.psi[:] - self.psi_axis)/(self.psi_bnd-self.psi_axis))
		self.psiN = equilibriumField((self.psi[:] - self.psi_axis)/(self.psi_bnd-self.psi_axis),psiN_func)

		R = [
		0.195, 0.195, 0.280, 0.280, 0.280, 0.175, 
		0.175, 0.190, 0.190, 0.330, 0.330, 0.535, 
		0.535, 0.755, 0.755, 0.755, 1.110, 1.655,
		1.655, 1.655, 2.000, 2.000, 1.975, 2.000, 
		2.000, 1.655, 1.655, 1.655, 1.110, 0.755, 
		0.755, 0.755, 0.535, 0.535, 0.330, 0.330,
		0.190, 0.190, 0.175, 0.175, 0.280, 0.280,
		0.280, 0.195, 0.195]
		
		Z = [
		0.000, 1.080, 1.220, 1.450, 1.670, 1.670,
		1.720, 1.820, 1.905, 2.150, 2.190, 2.190,
		2.165, 2.165, 1.975, 1.826, 1.826, 1.826,
		1.975, 2.165, 2.165, 0.300, 0.000,-0.300,
		-2.165,-2.165,-1.975,-1.826,-1.826,-1.826,
		-1.975,-2.165,-2.165,-2.190,-2.190,-2.150,
		-1.905,-1.820,-1.720,-1.670,-1.670,-1.450,
		-1.220,-1.080, 0.000]

		self.wall = { 'R' : R, 'Z' : Z }
		if with_bfield: self.calc_bfield()

	def load_JET(self,shot,time,sequence=0,with_bfield=True,verbose=False):
		""" Read in data from MAST IDAM

		arguments:
			shot = int	shot number to read in
			time = float	time to read in data at
		"""
			
		try:
			from ppf import ppfget,ppfgo,ppferr,ppfuid
		except:
			raise ImportError("No ppf module found, cannot load JET shot")

		if verbose: print("\nLoading equilibrium from JET shot "+str(shot)+"\n")
		
		
		ppfuid('JETPPF','r')

		self._shot = shot
		self._sequence = sequence
		self.machine = 'JET'

		def ppf_error_check(func,err):
			if err != 0:
				msg,ierr = ppferr(func,err)
				exit(func+": error code : "+msg)		

		err = ppfgo(self._shot,self._sequence)
		ppf_error_check('PPFGO',err)

		if not self._loaded:
			#Only store things we will need again as attributes
			(c_bnd  , m_bnd,   self._bnd_data, x_bnd,   t_bnd,   error_bnd)  = ppfget(shot, "EFIT", "fbnd")
			ppf_error_check('PPFGET',error_bnd)
			(c_axs  , m_axs,   self._axs_data, x_axs,   t_axs,   error_axs)  = ppfget(shot, "EFIT", "faxs")
			ppf_error_check('PPFGET',error_axs)
			(c_psi  , m_psi,   self._psi_data, x_psi,  self._t_psi,   error_psi)  = ppfget(shot, "EFIT", "psi")
			ppf_error_check('PPFGET',error_psi)
			(c_psir , m_psir , self._psi_r,    x_psir,  t_psir,  error_psir) = ppfget(shot, "EFIT", "psir")
			ppf_error_check('PPFGET',error_psir)
			(c_psiz , m_psiz , self._psi_z,    x_psiz,  t_psiz,  error_psiz) = ppfget(shot, "EFIT", "psiz")
			ppf_error_check('PPFGET',error_psiz)
			(c_rbphi, m_rbphi, self._r_bphi,   x_rbphi, t_rbphi, error_rbphi)= ppfget(shot, "EFIT", "bvac")
			ppf_error_check('PPFGET',error_rbphi)
			(c_xpointr, m_xpointr, self._xpointr, x_xpointr, t_xpointr, error_xpointr)= ppfget(shot, "EFIT", "rxpm")
			ppf_error_check('PPFGET',error_xpointr)
			(c_xpointz, m_xpointz, self._xpointz, x_xpointz, t_xpointz, error_xpointz)= ppfget(shot, "EFIT", "zxpm")
			ppf_error_check('PPFGET',error_xpointz)
			(c_axisr, m_axisr, self._axisr, x_axisr, t_axisr, error_axisr)= ppfget(shot, "EFIT", "rmag")
			ppf_error_check('PPFGET',error_axisr)
			(c_axisz, m_axisz, self._axisz, x_axisz, t_axisz, error_axisz)= ppfget(shot, "EFIT", "zmag")
			ppf_error_check('PPFGET',error_axisz)

		# Specify radius where toroidal field is specified
		self._r_at_bphi = 2.96
		
		self._psi_data = np.reshape(self._psi_data,(len(self._t_psi),len(self._psi_r),len(self._psi_z)))
		tind = np.abs(np.asarray(self._t_psi) - time).argmin()

		# Define the wall geometry
		wall_r = [3.2900,3.1990,3.1940,3.0600,3.0110,2.9630,2.9070,2.8880,2.8860,2.8900,2.9000, \
		          2.8840,2.8810,2.8980,2.9870,2.9460,2.8700,2.8340,2.8140,2.7000,2.5760,2.5550, \
		          2.5500,2.5220,2.5240,2.4370,2.4060,2.4180,2.4210,2.3980,2.4080,2.4130,2.4130, \
		          2.4050,2.3600,2.2950,2.2940,2.2000,2.1390,2.0810,1.9120,1.8210,1.8080,1.8860, \
		          1.9040,1.9160,2.0580,2.1190,2.1780,2.2430,2.3810,2.5750,2.8840,2.9750,3.1490, \
		          3.3000,3.3980,3.4770,3.5620,3.6400,3.6410,3.7430,3.8240,3.8855,3.8925,3.8480, \
		          3.7160,3.5780,3.3590,3.3090,3.2940,3.2900]

		wall_z = [-1.1520,-1.2090,-1.2140,-1.2980,-1.3350,-1.3350,-1.3830,-1.4230,-1.4760,-1.4980, \
		          -1.5100,-1.5820,-1.6190,-1.6820,-1.7460,-1.7450,-1.7130,-1.7080,-1.6860,-1.6510, \
		          -1.6140,-1.6490,-1.6670,-1.7030,-1.7100,-1.7110,-1.6900,-1.6490,-1.6020,-1.5160, \
		          -1.5040,-1.4740,-1.4290,-1.3860,-1.3340,-1.3340,-1.3200,-1.2440,-1.0970,-0.9580, \
		          -0.5130,-0.0230,0.4260,1.0730,1.1710,1.2320,1.6030,1.7230,1.8390,1.8940,1.9670, \
		           2.0160,1.9760,1.9410,1.8160,1.7110,1.6430,1.5720,1.4950,1.4250,1.2830,1.0700, \
		           0.8290,0.4950,0.2410,-0.0960,-0.4980,-0.7530,-1.0310,-1.0800,-1.1110,-1.1520]
		           
		# Interpolate the equilibrium onto a finer mesh
		psi_func = interp2d(self._psi_r,self._psi_z,self._psi_data[tind,:,:])
		self.psi = equilibriumField(self._psi_data[tind,:,:],psi_func)
		self.nr = len(self._psi_r)
		self.nz = len(self._psi_z)
		self.R = self._psi_r
		self.Z = self._psi_z
		
		self.psi_axis = self._axs_data[tind]
		self.psi_bnd  = self._bnd_data[tind] 
		self.Btcent   = self._r_bphi[tind]
		self.Rcent    = self._r_at_bphi
		self.sigBp    = 1.0

		self.nxpt = 1
		
		self.xpoint = [Point(self._xpointr.data[tind],self._xpointz.data[tind])]
		self.axis = Point(self._axisr.data[tind],self._axisz.data[tind])


		self.fpol = None
		
		tmp = (self.psi[:] - self.psi_axis)/(self.psi_bnd-self.psi_axis)

		psiN_func = interp2d(self.R,self.Z,(self.psi[:] - self.psi_axis)/(self.psi_bnd-self.psi_axis))
		self.psiN = equilibriumField((self.psi[:] - self.psi_axis)/(self.psi_bnd-self.psi_axis),psiN_func)


		self._loaded = True
		self._time = self._t_psi[tind]

		self.wall = { 'R' : wall_r, 'Z' : wall_z }
		if with_bfield: self.calc_bfield()

	def load_TCV(self,shot,time,port=None,with_bfield=True,verbose=False):
   		try:
   			import MDSplus as mds
   		except:
   		 	raise ImportError("Error: No MDSplus module found. \nPlease see www.mdsplus.org to install MDSplus\n")
   		
   		if port is None:
   		 	port = raw_input("Please enter port for ssh tunnel to TCV LAC: ")            
   		try:
   			conn = mds.Connection('localhost:'+str(port))
			self._mdsport = port
   		except:
   			print("---------------------------------------------------------------------")
   			print("Have you created an ssh tunnel to the TCV LAC?")
   			print("Create a tunnel by typing into the command line")
   			print("      ssh username@lac.911.epfl.ch -L XYZ:tcv1.epfl.ch:8000")
   			print("where XYV is your port number (i.e 1600)")
   			print("---------------------------------------------------------------------")
   			raise RuntimeError("Error: Could not connect through port")
        		
   		#Open tree to Lique data for equilibrium info
		
		self.machine = 'TCV'
   		if not self._loaded:
			conn.openTree('tcv_shot',shot)
   			#Load in required data from Lique off the MDS server
   			self._psi = conn.get('\\results::psi').data()
			self._time_array = conn.get('dim_of(\\results::psi)').data()
			nr = self._psi.shape[2]
			nz = self._psi.shape[1]
   			self._r = np.linspace(conn.get('\\results::parameters:ri').data(),conn.get('\\results::parameters:ro').data(),nr)	    
			z0 = conn.get('\\results::parameters:zu').data()
			self._z   = np.linspace(-z0,z0,nz)		  
			self._psi_axis = conn.get('\\results::psi_axis').data()
			self._psi_bnd  = conn.get('\\results::psi_xpts').data()
			self._bphi = conn.get('\\magnetics::rbphi').data()
			self._bt_time_array = conn.get('dim_of(\\magnetics::rbphi)').data()    
			self._xpointr = conn.get('\\results::r_xpts').data()  	   
			self._xpointz = conn.get('\\results::z_xpts').data()     	    
			self._axisr = conn.get('\\results::r_axis').data()  		  
			self._axisz = conn.get('\\results::z_xpts').data()  
			conn.closeTree('tcv_shot',shot)
			self._loaded = True
   		
   		tind = np.abs(self._time_array - time).argmin()		
		print "\n\n",tind,"\n\n"
		self.R = self._r#.data[0,:]
		self.Z = self._z#.data[0,:]
		psi_func = interp2d(self.R,self.Z,self._psi[tind])
		self.psi = equilibriumField(self._psi[tind],psi_func) 
		self.nr = len(self.R)
		self.nz = len(self.Z)		
		tind_ax = tind#np.abs(self._psi_axis.time - time).argmin()
		self.psi_axis = self._psi_axis[tind_ax]
		tind_bnd = tind#np.abs(self._psi_bnd.time - time).argmin()
		self.psi_bnd = 0.0#np.max(self._psi_bnd[tind_bnd])
		self.Rcent = 0.88 # Hard-coded(!)
		tind_Bt = np.abs(self._bt_time_array - time).argmin()
		self.Btcent = self._bphi[0,tind_Bt]
		tind_sigBp = tind#np.abs(self._cpasma.time - time).argmin()
		self.sigBp = -1.0#-(abs(self._cpasma.data[tind_sigBp])/self._cpasma.data[tind_sigBp])
	    		
		self.nxpt = 2
		tind_xpt = tind#np.abs(self._xpoint1r.time - time).argmin()
		self.xpoint = [Point(self._xpointr[tind_xpt],self._xpointz[tind_xpt])]
		#self.xpoint.append(Point(self._xpoint2r.data[tind_xpt],self._xpoint2z.data[tind_xpt]))
		self.axis = Point(self._axisr[tind_xpt],self._axisz[tind_xpt])		
		self.fpol = None		
        		
		self._loaded = True
		self._time = 0.0#self._psi.time[tind]
        		
		psiN_func = interp2d(self.R,self.Z,(self.psi[:] - self.psi_axis)/(self.psi_bnd-self.psi_axis))
		self.psiN = equilibriumField((self.psi[:] - self.psi_axis)/(self.psi_bnd-self.psi_axis),psiN_func)
        		
		R = [0.624,0.624,0.666,0.672,0.965,0.971,1.136,1.136,0.971,0.965,0.672,0.666,0.624,0.624,0.624]
		Z = [0.697,0.704,0.75,0.75,0.75,0.747,0.55,-0.55,-0.747,-0.75,-0.75,-0.75,-0.704,-0.697,0.697]
		
        		
		self.wall = { 'R' : R, 'Z' : Z }
		if with_bfield: self.calc_bfield()

	def set_time(self,time):
		if self._loaded and self.machine is not None:
			if self.machine == 'MAST':
				self.load_MAST(self._shot,time)
			elif self.machine == 'JET':
				self.load_JET(self._shot,time)
			elif self.machine == 'TCV':
				self.load_TCV(self._shot,time,port=self._mdsport)



	def plot_flux(self,col_levels=None,Nlines=25,axes=None):
		from copy import deepcopy as copy
		""" 
		Contour plot of the equilibrium poloidal flux

		keywords:
			t = int			time index to plot flux at
			col_levels = []		array storing contour levels for color plot
			Nlines = int		number of contour lines to display
		"""
		import matplotlib.pyplot as plt
		#Set contour levels		
		if self.psi is not None:
			if col_levels is None:
				col_levels = np.linspace(np.min(self.psi),np.max(self.psi),100)
			
			if axes is None:
				plt.contourf(self.R,self.Z,self.psi,levels=col_levels)
				cbar = plt.colorbar(format='%.3f')
				plt.contour(self.R,self.Z,self.psi,Nlines,colors='k')		
				plt.contour(self.R,self.Z,self.psi,[self.psi_bnd],colors='r',linewidth=1.5)
				plt.xlim(np.min(self.R),np.max(self.R))
				plt.ylim(np.min(self.Z),np.max(self.Z))
				if self.wall is not None:
					plt.plot(self.wall['R'],self.wall['Z'],'-k',linewidth=2.5)
				if self.xpoint:
					for xpoint in self.xpoint:
						plt.plot(xpoint.r,xpoint.z,'rx')
				if self.axis is not None:
					try:
						plt.plot(self.axis.r,self.axis.z,'ro')
					except:
						pass
				plt.xlabel('R (m)')
				plt.ylabel('Z (m)')
				cbar.ax.set_ylabel('$\psi$')
				plt.title('Equilibrium flux')
				plt.axes().set_aspect('equal')
				plt.show()
			else:
				axes.contourf(self.R,self.Z,self.psi,levels=col_levels)
				axes.contour(self.R,self.Z,self.psi,Nlines,colors='k')
				axes.contour(self.R,self.Z,self.psi,[self.psi_bnd],colors='r',linewidth=1.5)
				if self.xpoint:
					for xpoint in self.xpoint:
						plt.plot(xpoint.r,xpoint.z,'rx')
				if self.axis is not None:
					plt.plot(self.axis.r,self.axis.z,'ro')
		else:
			print("ERROR: No poloidal flux found. Please load an equilibrium before plotting.\n")
	
	
	def get_fluxsurface(self,psiN,Rref=1.5,Zref=0.0):
		"""
		Get R,Z coordinates of a flux surface at psiN
		"""
		try:
			import matplotlib.pyplot as plt
		except:
			print("ERROR: matplotlib required for flux surface construction, returning")
			return
			
		
		#axes = plt.axes([0.0,0.0,0.0,0.0])
		if type(psiN) is list:
			surfaces = []
			for psiNval in psiN:
				psi_cont = plt.contour(self.R,self.Z,(self.psi - self.psi_axis)/(self.psi_bnd-self.psi_axis),levels=[0,psiN],alpha=0.0)
				paths = psi_cont.collections[1].get_paths()
				#Figure out which path to use
				i = 0
				old_dist = 100.0
				for path in paths:
					dist = np.min(((path.vertices[:,0] - Rref)**2.0 + (path.vertices[:,1] - Zref)**2.0)**0.5)
					if  dist < old_dist:
						true_path = copy(path)
					old_dist = copy(dist) 

				R,Z =  true_path.vertices[:,0],true_path.vertices[:,1]
				surfaces.append(fluxSurface(R = R, Z = Z))
			return surfaces
		else:
			psi_cont = plt.contour(self.R,self.Z,(self.psi - self.psi_axis)/(self.psi_bnd-self.psi_axis),levels=[0,psiN],alpha=0.0)
			paths = psi_cont.collections[1].get_paths()
			old_dist = 100.0
			for path in paths:
				dist = np.min(((path.vertices[:,0] - Rref)**2.0 + (path.vertices[:,1] - Zref)**2.0)**0.5)
				if  dist < old_dist:
					true_path = path
				old_dist = dist
			R,Z =  true_path.vertices[:,0],true_path.vertices[:,1]
			#plt.clf()	
			return fluxSurface(R = R, Z = Z)
			
		
	
	
	
	def plot_var(self,var,title=None,col_levels=None):
		import matplotlib.pyplot as plt
		if col_levels==None:
			col_levels = np.linspace(np.min(var),np.max(var),100)
		plt.contourf(self.R,self.Z,var,levels=col_levels)
		cbar = plt.colorbar(format='%.3f')
			
		plt.xlabel('R (m)')
		plt.ylabel('Z (m)')
		if title != None:
			plt.title(title)
		plt.axes().set_aspect('equal')
		plt.show()
		
		
	def __calc_psi_deriv(self,method='CD'):
		"""
		Calculate the derivatives of the poloidal flux on the grid in R and Z
		"""
		if self.psi is None or self.R is None or self.Z is None:
			print("ERROR: Not enough information to calculate grad(psi). Returning.")
			return
		
		
		if method is 'CD':	#2nd order central differencing on an interpolated grid
			R = np.linspace(np.min(self.R),np.max(self.R),200)
			Z = np.linspace(np.min(self.Z),np.max(self.Z),200)
			Rgrid,Zgrid = R,Z #np.meshgrid(R,Z)
			psi  = self.psi(Rgrid,Zgrid)
			deriv = np.gradient(psi)    #gradient wrt index
			#Note np.gradient gives y derivative first, then x derivative
			ddR = deriv[1]
			ddZ = deriv[0]
			dRdi = 1.0/np.gradient(R)
			dZdi = 1.0/np.gradient(Z)
			dpsidR = ddR*dRdi[np.newaxis,:]	#Ensure broadcasting is handled correctly
			dpsidZ = ddZ*dZdi[:,np.newaxis]
			dpsidR_func = interp2d(R,Z,dpsidR)
			dpsidZ_func = interp2d(R,Z,dpsidZ)

			RR,ZZ = self.R,self.Z
			self.dpsidR = equilibriumField(np.transpose(dpsidR_func(RR,ZZ)),dpsidR_func)
			self.dpsidZ = equilibriumField(np.transpose(dpsidZ_func(RR,ZZ)),dpsidZ_func)
		else:
			print("ERROR: Derivative method not implemented yet, reverting to CD")
			self.calc_psi_deriv(method='CD')
			
			
	def calc_bfield(self):
				
		"""Calculate magnetic field components"""

		self.__calc_psi_deriv()		
			
		BR = self.dpsidZ/self.R[np.newaxis,:]
		BZ = self.sigBp*self.dpsidR/self.R[np.newaxis,:]
		Bp = self.sigBp*(BR**2.0 + BZ**2.0)**0.5	
			
		self.__get_fpolRZ()
		Bt = self.fpolRZ/self.R[np.newaxis,:]
		B = (BR**2.0 + BZ**2.0 + Bt**2.0)**0.5

		BR_func = interp2d(self.R,self.Z,BR)
		BZ_func = interp2d(self.R,self.Z,BZ)
		Bp_func = interp2d(self.R,self.Z,Bp)
		Bt_func = interp2d(self.R,self.Z,Bt)
		B_func = interp2d(self.R,self.Z,B)

		self.BR = equilibriumField(BR,BR_func)
		self.BZ = equilibriumField(BZ,BZ_func)
		self.Bp = equilibriumField(Bp,Bp_func)
		self.Bt = equilibriumField(Bt,Bt_func)
		self.B = equilibriumField(B,B_func)
		

	def __get_fpolRZ(self,plasma_response=False):
		""" 
		Generate fpol on the RZ grid given fpol(psi) and psi(RZ)
		fpol(psi) is given on an evenly spaced grid from psi_axis to psi_bnd. This means that
		some regions of psi will exceed the range that fpol is calculated over. 
		When this occurs (usually in the SOL) we assume that the plasma contribution
		is negligable and instead just take the vacuum assumption for the B-field and 
		reverse engineer
		
		"""
		from scipy.interpolate import interp1d
				
		fpolRZ = np.zeros((self.nz,self.nr))
		
		if plasma_response and self.fpol is not None:
			psigrid = np.linspace(self.psi_axis,self.psi_bnd,len(self.fpol))
			for i in np.arange(self.nr):
				for j in np.arange(self.nz):
					if self.psi[i,j] < psigrid[-1] and self.psi[i,j] > psigrid[0]:
						fpolRZ[i,j] = self.fpol(self.psi[i,j])
					else:
						fpolRZ[i,j] = self.Btcent*self.Rcent
		
		else:
			fpolRZ[:,:] = self.Btcent*self.Rcent
		
		fpolRZ_func = interp2d(self.R,self.Z,fpolRZ)
		self.fpolRZ = equilibriumField(fpolRZ,fpolRZ_func)
		

if __name__=='__main__':
	#Load a MAST shot 
	eq = equilibrium(device='TCV',shot=47651,time=0.3,mdsport=1600)
	
	#plot flux surfaces
	eq.plot_flux()
	
	#Change the shot time
	eq.set_time(1.2)
	
	#Replot flux surfaces
	eq.plot_flux()

	#plot B-field on efit grid
	eq.plot_var(eq.B)

	#plot B-field on refined grid
	RR,ZZ = np.linspace(np.min(eq.R),np.max(eq.R),100),np.linspace(np.min(eq.Z),np.max(eq.Z),200)
	import matplotlib.pyplot as plt
	plt.contourf(RR,ZZ,eq.B(RR,ZZ))
	plt.show()

	
	
