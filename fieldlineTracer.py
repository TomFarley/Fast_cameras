#!/usr/bin/env python
from __future__ import division

"""
Base class and subclassed implementations of field line tracers used to trace the 3D
trajectory of magnetic field lines from a given magnetic equilibrium.

Nick Walkden, May 2015
"""

import numpy as np
import sys
import matplotlib.pyplot as plt
from equilibrium import equilibrium 
from collections import namedtuple
from copy import deepcopy as copy
from scipy.interpolate import interp2d
from scipy.interpolate import UnivariateSpline as interp1d
#fieldline = namedtuple('fieldline','R Z phi X Y S')#' B bR bZ bt bp')

class fieldline(object):
	"""
	Simple class to hold information about a single field line.
	"""
	
	
	
	def __init__(self,**kwargs):
		"""
		R = Major Radius, Z = Vertical height, phi = toroidal angle
		X,Y,Z = Cartesian coordinates with origin at R=0
		S = Distance along field line
		B = magnetic field strength
		bR,bZ = magnetic field tangent vector components in R and Z
		bt,bp = magnetic field tangent vector components in toroidal and poloidal angle
		"""
		
		
		args = ['R','Z','phi','S','X','Y','B','bR','bt','bp','bR','bZ']		
		for key in args:
			if key in kwargs:
				super(fieldline,self).__setattr__(key,kwargs[key])

		
		

	def list(self):
		"""
		return a list of variable names stored in the class 
		"""
		return self.__dict__.keys()


	def rotate_toroidal(self,rotateAng=None):
		"""
		Rotate the field line by a given angle in the toroidal direction
		"""
		
		if rotateAng != None:
			#try:
			self.X = list(self.R*np.cos(np.array(self.phi) + rotateAng))
			self.Y = list(self.R*np.sin(np.array(self.phi) + rotateAng))
			self.phi = list(np.asarray(self.phi) + rotateAng)	
			#except:
				#raise AttributeError('ERROR: Check that R, Z and phi are defined')
		else:
			print("WARNING: No rotational angle given, returning")
			self.X = self.R*np.cos(self.phi)
			self.Y = self.R*np.sin(self.phi)
			
	


points = namedtuple('points','x y')

class fieldlineTracer:
	
	def __init__(self,**kwargs):
		""" initialize base class 
		
		keywords:
			shot = int		if given, read in data for shot on initialization
			gfile = file or str	if given, read in data from gfile
			tind = int		if reading from idam use tind
			machine = str		state which machine to read data for
		"""
		
		self.eq = equilibrium()		#Stored details of the efit equilibrium

		

		if 'shot' in kwargs and 'gfile' not in kwargs or kwargs['gfile'] is None:
			shot = kwargs['shot']
			if 'machine' not in kwargs:
				print("\nWARNING: No machine given, assuming MAST")
				machine = 'MAST'
			else:
				machine = kwargs['machine']
			if 'time' not in kwargs:
				print("\nWarning: No time given, setting to 0.25")
				time = 0.25
			else:
				time = kwargs['time']
			if 'mdsport' not in kwargs:
				mdsport = None
			else:
				mdsport = kwargs['mdsport']
				
			self.get_equilibrium(shot=shot,machine=machine,time=time,mdsport=mdsport)
		
			
		elif 'gfile' in kwargs:
			#If a gfile is found use it by default
			gfile = kwargs['gfile']
			if gfile != None:
				self.get_equilibrium(gfile)
	
		else:
			print("WARNING: No gfile, or shot number given, returning.")
			

		

	def get_equilibrium(self,gfile=None,shot=None,machine=None,time=None,mdsport=None):
		""" Read in data from efit

		keywords:
			machine = str	machine to load shot number for
		"""
		
		if gfile==None and shot != None:
			#load from IDAM
			if machine == 'MAST':
				self.eq.load_MAST(shot,time)
			elif machine == 'JET':
				self.eq.load_JET(shot,time)
			elif machine == 'TCV':
				self.eq.load_TCV(shot,time,mdsport)
		elif gfile != None:
			#load from gfile
			self.eq.load_geqdsk(gfile)
	

	def wall_intersection(self,A,B):
		"""
		Check for an intersection between the line (x1,y1),(x2,y2) and the wall
		"""
		
	      
		for i in np.arange(len(self.eq.wall['R'])-1):
			C = points(x=self.eq.wall['R'][i],y=self.eq.wall['Z'][i])
			D = points(x=self.eq.wall['R'][i+1],y=self.eq.wall['Z'][i+1])
			
			#First check for intersection
			def ccw(A,B,C):
				""" check if points are counterclockwise """
				return (C.y - A.y)*(B.x-A.x) > (B.y-A.y)*(C.x-A.x)
				
			if ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D): 
				
				#Find gradient of lines
				if A.x == B.x:
					Mab = 1e10*(B.y-A.y)/abs(B.y-A.y)
				else:
					Mab = (B.y - A.y)/(B.x - A.x)
				if C.x == D.x:
					Mcd = 1e10*(D.y-C.y)/abs(D.y-C.y)
				else:	
					Mcd = (D.y - C.y)/(D.x - C.x)
							
				#Find axis intercepts
				Cab = B.y - Mab*B.x
				Ccd = C.y - Mcd*C.x
				
				#Find line intersection point
				intr = points(x=(Ccd - Cab)/(Mab - Mcd),y=Cab + Mab*(Ccd - Cab)/(Mab - Mcd))
				
				return True , intr			

		return False,None	
	
	def set_psiN(self,psiN):
		pass
	
	def trace(self,Rstart,Zstart,phistart=0.0,mxstep=1000,ds=1e-3,backward=False,verbose=True):
		from scipy.interpolate import interp1d
		take_step = self.take_step
		
		if 'psiN' in self.__dict__:
			#Ensure that the starting points lie on the desired flux surface
			print("\n Refining starting position to ensure flux surface alignment for psi_N = "+str(self.psiN)+"\n")
			print("Original:\tRstart = "+str(Rstart)+"\tZstart = "+str(Zstart)+"\n")
			
			#Keep Zstart and find refined Rstart
			Rax = np.linspace(np.min(self.eq.R),np.max(self.eq.R),200)
			psi_R = self.eq.psiN(Rax,Zstart)
			
			#Now get R(psi)
			R_psi = interp1d(psi_R,Rax)
			Rstart = R_psi(self.psiN)
			
			print("Refined:\tRstart = "+str(Rstart)+"\tZstart = : "+str(Zstart)+"\n")
			
			if backward:
				self._currentS=0.0
			
		
		s = [0.0]
		R = [Rstart]
		Z = [Zstart]
		phi = [phistart]
		X = [Rstart*np.cos(phistart)]
		Y = [Rstart*np.sin(phistart)]
		i = 0
		while i < mxstep:
			if verbose:
				sys.stdout.flush()
				sys.stdout.write("Taking step number %d \r" %(i))
			dR,dZ,dphi = take_step(R=R[i],Z=Z[i],ds=ds)
			if self.eq.wall != None:
					A = points(x=R[i],y=Z[i])
					B = points(x=R[i]+dR,y=Z[i]+dZ)
					collision,intr = self.wall_intersection(A,B)
					if collision:
						if verbose: print("\nInterception with wall detected at R = "+str(intr.x)+" Z = "+str(intr.y)+"\n")
						R.append(intr.x)
						Z.append(intr.y)
						phicol = phi[i] + dphi*(intr.x-R[i])/dR
						phi.append(phicol)
						X.append(intr.x*np.cos(phicol))
						Y.append(intr.x*np.sin(phicol))
						s.append(s[i] + (ds/np.abs(ds))*((X[-1]-X[-2])**2.0 + (Y[-1] - Y[-2])**2.0 + (Z[-1] - Z[-2])**2.0)**0.5)
						if backward==True:
							break
						back = self.trace(Rstart,Zstart,phistart,mxstep,-ds,backward=True,verbose=verbose)
						R = back.R[::-1] + R[1:]
						Z = back.Z[::-1] + Z[1:]
						phi = back.phi[::-1] + phi[1:]
						X = back.X[::-1] + X[1:]
						Y = back.Y[::-1] + Y[1:]
						s = back.S[::-1] + s[1:]
						
						break
			
			R.append(copy(R[i] + dR))
			Z.append(copy(Z[i] + dZ))
			phi.append(copy(phi[i]+dphi))			
			X.append(R[i+1]*np.cos(phi[i+1]))
			Y.append(R[i+1]*np.sin(phi[i+1]))		
			s.append(s[i] + ds)
		
			i += 1
	
		return fieldline(R=R,Z=Z,phi=phi,X=X,Y=Y,S=s)#,B=B(R,Z),bR=bR(R,Z),bZ=bZ(R,Z),bt=bt(R,Z),bp=bp(R,Z))
		
		
		

class RK4Tracer(fieldlineTracer):
	"""
	RK4 field line tracer
	"""		
	
	def __init__(self,**kwargs):
		fieldlineTracer.__init__(self,**kwargs)
		from scipy.interpolate import interp2d
		
		if 'interp' not in kwargs or kwargs['interp'] is None:
			interp = 'linear'
		else:
			interp = kwargs['interp']
		if self.eq.B is None:
			self.eq.calc_bfield()
			
		self._bR = interp2d(self.eq.R,self.eq.Z,self.eq.BR/self.eq.B,kind=interp)
		self._bZ = interp2d(self.eq.R,self.eq.Z,self.eq.BZ/self.eq.B,kind=interp)
		self._bt = interp2d(self.eq.R,self.eq.Z,self.eq.Bt/self.eq.B,kind=interp)


	def take_step(self,R,Z,ds):
		"""
		Take an RK4 step along the field line
		
		R,Z	starting values
		ds	step size
		bR,bZ,bphi	Magnetic field functions that must be able to perform F(R,Z) = num 
		bpol not used here	
		"""
		bR = self._bR
		bZ = self._bZ
		bt = self._bt

		dR1 = ds*bR(R,Z)
		dZ1 = ds*bZ(R,Z)
		dphi1 = ds*bt(R,Z)/R
		
		dR2 = ds*bR(R+0.5*dR1,Z+0.5*dZ1)
		dZ2 = ds*bZ(R+0.5*dR1,Z+0.5*dZ1)
		dphi2 = ds*bt(R+0.5*dR1,Z+0.5*dZ1)/R
		
		dR3 = ds*bR(R+0.5*dR2,Z+0.5*dZ2)
		dZ3 = ds*bZ(R+0.5*dR2,Z+0.5*dZ2)
		dphi3 = ds*bt(R+0.5*dR2,Z+0.5*dZ2)/R
		
		dR4 = ds*bR(R+dR3,Z+dZ3)
		dZ4 = ds*bZ(R+dR3,Z+dZ3)
		dphi4 = ds*bt(R+dR3,Z+dZ3)/R
		
		dR = (1./6.)*(dR1 + 2.0*dR2 + 2.0*dR3 + dR4)
		dZ = (1./6.)*(dZ1 + 2.0*dZ2 + 2.0*dZ3 + dZ4)
		dphi = (1./6.)*(dphi1 + 2.0*dphi2 + 2.0*dphi3 + dphi4)
		
		return float(dR),float(dZ),float(dphi)
		

		
		
class EulerTracer(fieldlineTracer):
	"""
	Forward Euler field line tracer (for benchmarking purposes ONLY)
	"""		
	
	def __init__(self,**kwargs):
		fieldlineTracer.__init__(self,**kwargs)
			
		if self.eq.B is None:
			self.eq.calc_bfield()
			
		if 'interp' not in kwargs:
			interp = 'linear'
		else:
			interp = kwargs['interp']
			
		self._bR = interp2d(self.eq.R,self.eq.Z,self.eq.BR/self.eq.B,kind=interp)
		self._bZ = interp2d(self.eq.R,self.eq.Z,self.eq.BZ/self.eq.B,kind=interp)
		self._bt = interp2d(self.eq.R,self.eq.Z,self.eq.Bt/self.eq.B,kind=interp)
		
	def take_step(self,R,Z,ds):
		"""
		Take a forward Euler step along the field line
		
		R,Z	starting values
		dl	step size
		bR,bZ,bphi	Magnetic field functions that must be able to perform F(R,Z) = num 
		bpol not used here	
		"""
		dR = ds*self._bR(R,Z)
		dZ = ds*self._bZ(R,Z)
		dphi = ds*self._bt(R,Z)/R
		
		return float(dR),float(dZ),float(dphi)
		

class fluxsurfaceTracer(fieldlineTracer):
	"""
	Trace a field line constrained to an equilibrium flux surface
	
	Steps along a field line but ensures that the field line remains on the desired flux-surface
	
	This is curretly limited by the resolution of the equilibrium flux-surface location
	
	"""		
	
	
	def __init__(self,**kwargs):
		fieldlineTracer.__init__(self,**kwargs)	
		if 'interp' not in kwargs:
			interp = 'linear'
		else:
			interp = kwargs['interp']
				
		if 'psiN' in kwargs and kwargs['psiN'] != None:
			self.psiN = kwargs['psiN']
		else:
			self.psiN = None	

		
		if self.eq.B is None:
			self.eq.calc_bfield()
			
		
			
		self._Bt = interp2d(self.eq.R,self.eq.Z,self.eq.Bt,kind=interp)
		self._Bp = interp2d(self.eq.R,self.eq.Z,self.eq.Bp,kind=interp)
		
		self._initialized=False
		
	def set_psiN(self,psiN):
		
		self.psiN = psiN
		self.FS = self.eq.get_fluxsurface(self.psiN,self._Rstart,self._Zstart)
		self._initialized = False
				
	def init(self,Rstart,Zstart):
		"""
		initialization that cannot be contained in __init__
		"""
		self._Rstart = Rstart
		self._Zstart = Zstart	

		if self.psiN is None:
			self.psiN = float(raw_input("Enter normalized psi of flux-surface"))
		self.set_psiN(self.psiN)
		
		#Get some interpolation functions for use later
		#when stepping along the field line
		phi = [0.0]
		L = [0.0]
		S = [0.0]
		
		for i in np.arange(len(self.FS.R)-1):
			
			#Shift in distance along flux surface in poloidal direction
			dl = ((self.FS.R[i+1] - self.FS.R[i])**2.0 + (self.FS.Z[i+1] - self.FS.Z[i])**2.0)**0.5
			
			#Corresponding shift along field line
			ds = self._Bt(self.FS.R[i+1],self.FS.Z[i+1])*dl/(self._Bp(self.FS.R[i+1],self.FS.Z[i+1]))
			
			#Shift in toroidal angle
			dphi = ds/self.FS.R[i+1]
			
			L.append(L[i] + dl)
			S.append(S[i] + ds)
			phi.append(phi[i] + dphi)
			
			
		#Find the starting point of S
		def sign_change(p1,p2):
			if p1*p2/np.abs(p1*p2) < 0:
				return True
			else:
				return False
				
		i = 0
		
		#Rtest = self.FS.R - Rstart
		Ztest = np.abs(np.asarray(self.FS.Z) - Zstart)
		i = Ztest.argmin()
		
			
		dl = ((Rstart - self.FS.R[i])**2.0 + (Zstart - self.FS.Z[i])**2.0)**0.5
		ds = self._Bt(Rstart,Zstart)*dl/(self._Bp(Rstart,Zstart))
		
		S -= S[i] + ds
			
		#Now generate R,Z,phi as a function of S
		
		
		self._Rs = interp1d(S,self.FS.R,s=0.0005)
		self._Zs = interp1d(S,self.FS.Z,s=0.0005)
		self._phis = interp1d(S,phi)
		 
		self._initialized = True
		self._currentS = 0.0
		
	def take_step(self,R,Z,ds):
		
		if not self._initialized:
			self.init(R,Z)
			
		dR = self._Rs(self._currentS + ds) - R
		dZ = self._Zs(self._currentS + ds) - Z
		dphi = self._phis(self._currentS + ds) - self._phis(self._currentS)
		
		self._currentS += ds
		
		return float(dR),float(dZ),float(dphi)
			
			
			
			
			
def get_fieldline_tracer(type,**kwargs):
	if type == 'Euler':
		return EulerTracer(**kwargs)
	elif type == 'RK4':
		return RK4Tracer(**kwargs)
	elif type == 'fluxsurface':
		return fluxsurfaceTracer(**kwargs)
		
	else:
		raise TypeError("No tracer of type ",type," implemented")
		return	
			
			
			
		
			
	
	
