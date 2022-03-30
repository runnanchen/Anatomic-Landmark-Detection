#!/usr/bin/env python

import sys
import math
import numpy as np

class Point:
	def __init__(self, x, y):
		self.x = x
		self.y = y
	def __str__(self):
		return str(self.x) + "," + str(self.y)

class Vector:
	def __init__(self, pa, pb):
		self.x = int(pb.x) - int(pa.x)
		self.y = int(pb.y) - int(pa.y)
	def __str__(self):
		return str(self.x) + "," + str(self.y)

class Angle:
	def __init__(self, va, vb):
		self.va = va
		self.vb = vb
	def theta(self):
		theta = math.degrees(math.acos(( self.va.x*self.vb.x + self.va.y*self.vb.y ) / (math.hypot(self.va.x, self.va.y)*math.hypot(self.vb.x, self.vb.y))))
		return theta

class Distance:
	def __init__(self, pa, pb):
		self.x = (int(pb.x) - int(pa.x))*(int(pb.x) - int(pa.x))
		self.y = (int(pb.y) - int(pa.y))*(int(pb.y) - int(pa.y))
	def dist(self):
		return  (self.x+self.y)**0.5

def checkArg():
	if len(sys.argv) != 2:
		print("please give me file")
		sys.exit(0)

def readFile(filename):
	points = []
	f = open(filename, "r")
	for line in f.readlines():
		line = line.strip(" \t\n\r")
		x = line.split(",")[0]
		y = line.split(",")[1]
		points.append(Point(x,y))
	f.close()
	return points

def getCross(va,vb):
	return va.x*vb.y - va.y*vb.x


def getODI(pa,pb,pc,pd,pe,pf,pg,ph):
	va = Vector(pa,pb)
	vb = Vector(pc,pd)
	vc = Vector(pe,pf)
	vd = Vector(pg,ph)

	aa = Angle(va,vb).theta()
	ab = Angle(vc,vd).theta()
	cb = getCross(vc,vd)
	#print cb
	if cb < 0:
		ab = -ab
	#print "u=" + str(aa)
	#print "v=" + str(ab)
	return aa+ab

def getAPDI(pa,pb,pc,pd,pe,pf,pg,ph,pi,pj):
	va = Vector(pa,pb)
	vb = Vector(pc,pd)
	vc = Vector(pe,pf)
	vd = Vector(pg,ph)
	ve = Vector(pi,pj)
	#print vb
	#print vc
	aa = Angle(va,vb).theta()
	ab = Angle(vb,vc).theta()
	ac = Angle(vd,ve).theta()

	cb = getCross(vb,vc)
	cc = getCross(vd,ve)
	#print cb
	if cb > 0:
		ab = -ab
	if cc < 0:
		ac = -ac
	#print "p=" + str(aa)
	#print "q=" + str(ab)
	#print "v=" + str(ac)
	return aa+ab+ac


def writeFile(filename,points,ANBtype,SNBtype,SNAtype,ODItype,APDItype,FHItype,FMAtype,mwtype):
	f = open(filename, "w")
	for point in points:
		f.write(str(point)+ "\n")
	f.write(ANBtype + "\n")
	f.write(SNBtype + "\n")
	f.write(SNAtype + "\n")
	f.write(ODItype + "\n")
	f.write(APDItype + "\n")
	f.write(FHItype + "\n")
	f.write(FMAtype + "\n")
	f.write(mwtype + "\n")
	f.close()



def classification(points):
	
	#points = readFile(filename)

	va = Vector(points[1],points[0])
	vb = Vector(points[1],points[5])
	vc = Vector(points[1],points[0])
	vd = Vector(points[1],points[4])
	#print "1. ANB" 
	ANBtype = ''
	ANB = Angle(vc,vd).theta() - Angle(va,vb).theta()
	#print ANB
	if ANB < 3.2:
		ANBtype='3'
	elif ANB > 5.7:
		ANBtype='2'
	else:
		ANBtype='1'
	#print ANBtype
	#print
	va = Vector(points[1],points[0])
	vb = Vector(points[1],points[5])
	#print "2. SNB"
	SNBtype = ''
	SNB = Angle(va,vb).theta()
	if SNB < 74.6:
		SNBtype='2'
	elif SNB > 78.7:
		SNBtype='3'
	else:
		SNBtype='1'
	#print SNB
	#print SNBtype
	#print
	va = Vector(points[1],points[0])
	vb = Vector(points[1],points[4])
	#print "3. SNA"
	SNAtype = ''
	SNA = Angle(va,vb).theta()
	if SNA < 79.4:
		SNAtype='3'
	elif SNA > 83.2:
		SNAtype='2'
	else:
		SNAtype='1'
	#print SNA
	#print SNAtype
	#print

	
	#print "4. ODI"
	ODItype = ''
	ODI = getODI(points[7],points[9],points[5],points[4],points[3],points[2],points[16],points[17])
	if ODI < 68.4:
		ODItype='3'
	elif ODI > 80.5:
		ODItype='2'
	else:
		ODItype='1'
	#print ODI
	#print ODItype
	#print
	
	#print "5. APDI"
	APDItype =''
	APDI = getAPDI(points[2],points[3],points[1],points[6],points[4],points[5],points[3],points[2],points[16],points[17])
	if APDI < 77.6:
		APDItype='2'
	elif APDI > 85.2:
		APDItype='3'
	else:
		APDItype='1'
	#print APDI
	#print APDItype
	#print
	
	pfh = Distance(points[0],points[9]).dist()
	afh = Distance(points[1],points[7]).dist()
	#print "6. FHI"
	FHItype = ''
	#print pfh/afh
	if pfh/afh < 0.65:
		FHItype='3'
	elif pfh/afh > 0.75:
		FHItype='2'
	else:
		FHItype='1'
	
	#print FHItype
	#print
	
	va = Vector(points[0],points[1])
	vb = Vector(points[9],points[8])
	#print "7. FMA"
	FMAtype = ''
	if Angle(va,vb).theta() < 26.8:
		FMAtype='3'
	elif Angle(va,vb).theta() > 31.4:
		FMAtype='2'
	else:
		FMAtype='1'
		
	#print Angle(va,vb).theta()
	#print FMAtype
	#print
	mw = Distance(points[10],points[11]).dist()/10
	mwtype = '';
	if points[11].x < points[10].x:
		mw = -mw
	if mw >= 2:
		if mw <= 4.5:
			mwtype = '1'
		else:
			mwtype = '4'
	elif mw == 0:
		mwtype = '2'
	else:
		mwtype = '3'
	#print "8. MW"
	#print mw
	#print mwtype

	
	#filename = "out-" + filename

	#writeFile(filename,points,ANBtype,SNBtype,SNAtype,ODItype,APDItype,FHItype,FMAtype,mwtype)
	return np.array([ANBtype,SNBtype,SNAtype,ODItype,APDItype,FHItype,FMAtype,mwtype])
