#!/usr/bin/env python3

import os
import sys
import glob

import shutil
import ntpath
import pickle
import datetime
import argparse
import numpy as np
import numpy.matlib

from scipy.interpolate import griddata
from scipy.signal import convolve2d as conv2
from xml.etree.ElementTree import ElementTree


import isce
import isceobj
from isceobj.Image import createImage,createDemImage
from imageMath import IML



def read_bands(filename, length, width, scheme, nbands, datatype):

    if datatype.upper() == 'FLOAT':
        datatype1 = np.float32
    elif datatype.upper() == 'CFLOAT':
        datatype1 = np.complex64
    elif datatype.upper() == 'DOUBLE':
        datatype1 = np.float64
    elif datatype.upper() == 'BYTE':
        datatype1 = np.int8
    elif datatype.upper() == 'SHORT':
        datatype1 = np.int16
    else:
        raise Exception('data type not supported, please contact crl for your data type!')

    bands = []
    if scheme.upper() == 'BIP':
        data = np.fromfile(filename, dtype=datatype1).reshape(length, width*nbands)
        for i in range(nbands):
            bands.append(data[:, i:width*nbands:nbands])
    elif scheme.upper() == 'BIL':
        data = np.fromfile(filename, dtype=datatype1).reshape(length*nbands, width)
        for i in range(nbands):
            bands.append(data[i:length*nbands:nbands, :])
    elif scheme.upper() == 'BSQ':
        data = np.fromfile(filename, dtype=datatype1).reshape(length*nbands, width)
        for i in range(nbands):
            offset = length * i
            bands.append(data[0+offset:length+offset, :])        
    else:
        raise Exception('unknown band scheme!')

    return bands


def write_bands(filename, length, width, scheme, nbands, datatype, bands):

    if datatype.upper() == 'FLOAT':
        datatype1 = np.float32
    elif datatype.upper() == 'CFLOAT':
        datatype1 = np.complex64
    elif datatype.upper() == 'DOUBLE':
        datatype1 = np.float64
    elif datatype.upper() == 'BYTE':
        datatype1 = np.int8
    elif datatype.upper() == 'SHORT':
        datatype1 = np.int16
    else:
        raise Exception('data type not supported, please contact crl for your data type!')

    if scheme.upper() == 'BIP':
        data = np.zeros((length, width*nbands), dtype=datatype1)
        for i in range(nbands):
            data[:, i:width*nbands:nbands] = bands[i]
    elif scheme.upper() == 'BIL':
        data = np.zeros((length*nbands, width), dtype=datatype1)
        for i in range(nbands):
            data[i:length*nbands:nbands, :] = bands[i]
    elif scheme.upper() == 'BSQ':
        data = np.zeros((length*nbands, width), dtype=datatype1)
        for i in range(nbands):
            offset = length * i
            data[0+offset:length+offset, :] = bands[i]
    else:
        raise Exception('unknown band scheme!')    

    #output result
    data.astype(datatype1).tofile(filename)

def write_bands(filename, length, width, scheme, nbands, datatype, bands):

    if datatype.upper() == 'FLOAT':
        datatype1 = np.float32
    elif datatype.upper() == 'CFLOAT':
        datatype1 = np.complex64
    elif datatype.upper() == 'DOUBLE':
        datatype1 = np.float64
    elif datatype.upper() == 'BYTE':
        datatype1 = np.int8
    elif datatype.upper() == 'SHORT':
        datatype1 = np.int16
    else:
        raise Exception('data type not supported, please contact crl for your data type!')

    if scheme.upper() == 'BIP':
        data = np.zeros((length, width*nbands), dtype=datatype1)
        for i in range(nbands):
            data[:, i:width*nbands:nbands] = bands[i]
    elif scheme.upper() == 'BIL':
        data = np.zeros((length*nbands, width), dtype=datatype1)
        for i in range(nbands):
            data[i:length*nbands:nbands, :] = bands[i]
    elif scheme.upper() == 'BSQ':
        data = np.zeros((length*nbands, width), dtype=datatype1)
        for i in range(nbands):
            offset = length * i
            data[0+offset:length+offset, :] = bands[i]
    else:
        raise Exception('unknown band scheme!')    

    #output result
    data.astype(datatype1).tofile(filename)



def runCmd(cmd):
    print("{}".format(cmd))
    status = os.system(cmd)
    if status != 0:
        raise Exception('error when running:\n{}\n'.format(cmd))


def getWidth(xmlfile):
    xmlfp = None
    try:
        xmlfp = open(xmlfile,'r')
        print('reading file width from: {0}'.format(xmlfile))
        xmlx = ElementTree(file=xmlfp).getroot()
        tmp = xmlx.find("component[@name='coordinate1']/property[@name='size']/value")
        if tmp == None:
            tmp = xmlx.find("component[@name='Coordinate1']/property[@name='size']/value")
        width = int(tmp.text)
        print("file width: {0}".format(width))
    except (IOError, OSError) as strerr:
        print("IOError: %s" % strerr)
        return []
    finally:
        if xmlfp is not None:
            xmlfp.close()
    return width


def getLength(xmlfile):
    xmlfp = None
    try:
        xmlfp = open(xmlfile,'r')
        print('reading file length from: {0}'.format(xmlfile))
        xmlx = ElementTree(file=xmlfp).getroot()
        tmp = xmlx.find("component[@name='coordinate2']/property[@name='size']/value")
        if tmp == None:
            tmp = xmlx.find("component[@name='Coordinate2']/property[@name='size']/value")
        length = int(tmp.text)
        print("file length: {0}".format(length))
    except (IOError, OSError) as strerr:
        print("IOError: %s" % strerr)
        return []
    finally:
        if xmlfp is not None:
            xmlfp.close()
    return length


def create_xml(fileName, width, length, fileType):
    
    if fileType == 'slc':
        image = isceobj.createSlcImage()
    elif fileType == 'int':
        image = isceobj.createIntImage()
    elif fileType == 'amp':
        image = isceobj.createAmpImage()
    elif fileType == 'rmg':
        image = isceobj.Image.createUnwImage()
    elif fileType == 'float':
        image = isceobj.createImage()
        image.setDataType('FLOAT')

    image.setFilename(fileName)
    image.setWidth(width)
    image.setLength(length)
        
    image.setAccessMode('read')
    #image.createImage()
    image.renderVRT()
    image.renderHdr()
    #image.finalizeImage()


def create_amp(width, length, master, slave, amp):
    amp_data = np.zeros((length, width*2), dtype=np.float)
    amp_data[:, 0:width*2:2] = np.absolute(master) * (np.absolute(slave)!=0)
    amp_data[:, 1:width*2:2] = np.absolute(slave) * (np.absolute(master)!=0)
    amp_data.astype(np.float32).tofile(amp)
    create_xml(amp, width, length, 'amp')
    
def geocoding(inputf,outputf,inps):
    inImage = createImage()
    inImage.load(inputf+'.xml')
    inImage.filename = inputf

    width = inImage.width
    length = inImage.length
    scheme = inImage.scheme
    nbands = inImage.bands
    print("number of bands : {}".format(nbands))
    datatype = inImage.dataType

    latImage = createImage()
    latImage.load(inps.lat+'.xml')
    latImage.filename = inps.lat

    lonImage = createImage()
    lonImage.load(inps.lon+'.xml')
    lonImage.filename = inps.lon

    if width != latImage.width or width != lonImage.width:
        raise Exception('file width are different!')
    if length != latImage.length or length != latImage.length:
        raise Exception('file length are different!')

    #convert to degrees
    sample_size = inps.ssize/3600.0
    sample_size_lat = sample_size
    sample_size_lon = sample_size
    print("geocoding sample size: {} [arcsec]".format(inps.ssize))
    print("geocoding sample size: {} [degree]".format(sample_size))

    if inps.rmethod == 0:
        rmethod = 'nearest'
    elif inps.rmethod == 1:
        rmethod = 'linear'
    elif inps.rmethod == 2:
        rmethod = 'cubic'
    else:
        raise Exception('unknow resampling method!')
    print("interpolation method: {}".format(rmethod))
    if inps.bbox:
        bbox= [float(val) for val in inps.bbox.split(',')]
        bbox= (bbox[0],bbox[1],bbox[2],bbox[3])
    else:
        bbox = (0,0,0,0)

    print("reading data to be geocoded")
    bands = read_bands(inputf, length, width, scheme, nbands, datatype)

    #get latitude
    print("reading latitude")
    lat = read_bands(inps.lat, length, width, latImage.scheme, latImage.bands, latImage.dataType)
    lat = lat[0]
    lat2= np.asarray(lat)
    lat2[lat2==0]=np.nan
    lat2=np.reshape(lat2,[length,width])
    win2=np.ones((5,5))/25
    lat2m=conv2(lat2,win2,'same')
    lat2m=lat2m.flatten()
    lat2=lat2.reshape(length*width)

    #get longitude
    print("reading longitude")
    lon = read_bands(inps.lon, length, width, lonImage.scheme, lonImage.bands, lonImage.dataType)
    lon = lon[0]
    lon2=lon
    lon2[lon2==0]=np.nan

    latlon = np.zeros((length*width, 2), dtype=np.float64)
    lat=lat.reshape(length*width)
    lon=lon.reshape(length*width)
    diffll=np.abs(lat2-lat2m)
    diffll[np.isnan(diffll)]=0
    lat2[lat2==0]=np.nan
    if bbox[0]==0:   
        mask= (diffll>0.1) | (np.isnan(lat2)) | ( np.isnan(lat2m)) | ( np.isnan(bands[0].reshape(length*width)))
    else:
        latlonmask= (lat>bbox[0]) & (lat<bbox[1]) & (lon>bbox[2]) & (lon<bbox[3]) 
        mask= (diffll>0.1) | (np.isnan(lat2)) | ( np.isnan(lat2m)) | ( np.isnan(bands[0].reshape(length*width))) |np.invert(latlonmask)
    lat=lat[mask==0] 
    lon=lon[mask==0] 
    if bbox[0]==0 and bbox[1]==0 and bbox[2]==0 and bbox[3]==0:
        bbox = ( np.nanmin(lat), np.nanmax(lat), np.nanmin(lon), np.nanmax(lon) )
    print("geocode bounding box:")
    print("south: {}".format(bbox[0]))
    print("north: {}".format(bbox[1]))
    print("west: {}".format(bbox[2]))
    print("east: {}".format(bbox[3]))


    latlon = np.zeros( (len(lat), 2), dtype=np.float64)
    latlon[:, 0] = lat
    latlon[:, 1] = lon
    bands2=np.zeros( (nbands, len(lat)), dtype=np.float32)
    for i in range(nbands):
       temp=bands[i].reshape(length*width)
       bands2[i]=temp[mask==0]
    #                             n       s                         w       e
    grid_lat, grid_lon = np.mgrid[bbox[1]:bbox[0]:-sample_size_lat, bbox[2]:bbox[3]:sample_size_lon]

    msk=np.zeros([length,width])
    msk[0:2,:]=1
    msk[-2:,:]=1
    msk[:,-2:]=1
    msk[:,0:2]=1
    msk= msk.reshape(length*width)
    msk=msk[mask==0]
    geomsk = griddata(latlon, msk, (grid_lat, grid_lon), method=rmethod, fill_value = 0.0)
    print("interpolate input data")
    bands_geo = []
    for i in range(nbands):
        geoband = griddata(latlon, (bands2[i]), (grid_lat, grid_lon), method=rmethod, fill_value = 0.0)
        bands_geo.append(np.multiply(geoband,(geomsk==0) ))

    print("write result")
    (length_geo, width_geo) = geoband.shape
    write_bands(outputf, length_geo, width_geo, scheme, nbands, datatype, bands_geo)


    outImage = inImage
    outImage.setFilename(outputf)
    outImage.setWidth(width_geo)
    outImage.setLength(length_geo)
    outImage.coord2.coordDescription = 'Latitude'
    outImage.coord2.coordUnits = 'degree'
    outImage.coord2.coordStart = bbox[1]
    outImage.coord2.coordDelta = -sample_size_lat
    outImage.coord1.coordDescription = 'Longitude'
    outImage.coord1.coordUnits = 'degree'
    outImage.coord1.coordStart = bbox[2]
    outImage.coord1.coordDelta = sample_size_lon
    outImage.renderHdr()

    cmdline="gdal_translate {}    {}".format(outputf + ".vrt",outputf + ".tif")
    os.system(cmdline)   



def cmdLineParse():
    '''
    Command line parser.
    '''
    parser = argparse.ArgumentParser( description='slcstk2gamp')
    parser.add_argument('-i', dest='slcdir', type=str, required=True,
            help = 'merged slc directory containing all scenes ex) merged/SLC')
    parser.add_argument('-o', dest='outdir', type=str, required=True,
            help = 'output directory')
    parser.add_argument('-lat', dest='lat', type=str, required=True,
            help = 'latitude file.')
    parser.add_argument('-lon', dest='lon', type=str, required=True,
            help = 'longitude file')
    parser.add_argument('-r',dest='rlks', type=int, default=0,
            help = 'number of range looks')
    parser.add_argument('-a',dest='alks', type=int, default=0,
            help = 'number of azimuth looks')
    parser.add_argument('-ssize', dest='ssize', type=float, default=1.0,
            help = 'output geocoded sample size. default: 1.0 arcsec')
    parser.add_argument('-rmethod', dest='rmethod', type=int, default=1,
            help = 'resampling method. 0: nearest. 1: linear (default). 2: cubic.')
    parser.add_argument('-bbox', dest='bbox', type=str, required=False,
            help='geocode bounding box (format: S,N,W,E). or you can input master frame pickle file that contains a bounding box')
    return parser.parse_args()


if __name__ == '__main__':

    #SCR_DIR="$INSAR_ZERODOP_SCR"
    SCR_DIR="$slccor/script/"

    inps = cmdLineParse()

    mbursts = sorted(glob.glob(os.path.join(os.path.abspath(inps.slcdir), '*/*.slc.full')))
    
    if not os.path.exists(inps.outdir):
        os.mkdir(inps.outdir)
   

    nb = len(mbursts) #number of slc scenes

    for i in range(nb):
        print('+++++++++++++++++++++++++++++++++++')
        print('processing scene {} of {}'.format(i+1, nb))
        print('+++++++++++++++++++++++++++++++++++')

        mslc = ntpath.basename(mbursts[i])
        print(mslc)
        slcdate=mslc.rstrip('\n').replace('*',"").split(".")[0]
        print(slcdate)
        
        outputampdir= inps.outdir + '/' + slcdate
        
        if not os.path.exists(outputampdir):
            os.mkdir(outputampdir)
               
        width = getWidth(mbursts[i] + '.xml')
        length = getLength(mbursts[i] + '.xml')

        width_looked = int(width/inps.rlks)
        length_looked = int(length/inps.alks)
        if not os.path.exists(os.path.join(outputampdir,slcdate + ".amp")): 
            cmd= "imageMath.py -e='abs(a)' --a={} -t FLOAT -s BIL -o {}".format(mbursts[i],
                 os.path.join(outputampdir,slcdate + ".amp"))
            runCmd(cmd)
        
        lookedname=os.path.join(outputampdir,slcdate + ".amp.{}r{}a".format(inps.rlks,inps.alks))

        if not os.path.exists(lookedname): 
            cmd= "looks.py --input {} --output {} -r {} -a {}".format(
                os.path.join(outputampdir,slcdate + ".amp"),
                lookedname,
                inps.rlks,
                inps.alks)
            runCmd(cmd)
            temp_name = os.path.join(outputampdir, slcdate, '.amp')
            os.remove(temp_name)
            os.rename(lookedname, temp_name)
            lookedname=temp_name
            
        output=lookedname + '.geo'
        
        if not os.path.exists(output):
            geocoding(lookedname,output,inps)

        outputdb=lookedname + '.db.geo'
        if not os.path.exists(outputdb):
            cmd= "imageMath.py -e='20*log10(a)' --a={} -t FLOAT -o {}".format(
                output,
                outputdb)
            runCmd(cmd)
            
            cmdline="gdal_translate {}    {}".format(outputdb + ".vrt",outputdb + ".tif")
            os.system(cmdline)   
        

#        master = np.fromfile(mbursts[i], dtype=np.complex64).reshape(length, width)
	

#        ampFile = 'amp_%02d.amp' % (i+1)
#        create_amp(width, length, master, slave, ampFile)

#        ampLookedFile = 'b%02d_%dr%dalks.amp' % (i+1,inps.rlks,inps.alks)
#        cmd = "{}/look.py -i {} -o {} -r {} -a {}".format(SCR_DIR,
#            ampFile, 
#            ampLookedFile,
#            inps.rlks,
#            inps.alks)
#        runCmd(cmd)



#        latMax = np.amax(latLooked)
#        latMin = np.amin(latLooked)
#        lonMax = np.amax(lonLooked)
#        lonMin = np.amin(lonLooked)
#        bbox = "{}/{}/{}/{}".format(latMin, latMax, lonMin, lonMax)

#        corLookedGeoFile = 'b%02d_%dr%dalks.cor.geo' % (i+1,inps.rlks,inps.alks)
#        cmd = "{}/geo_with_ll.py -input {} -output {} -lat {} -lon {} -bbox {} -ssize {} -rmethod {}".format(SCR_DIR,
#        cmd = "{}/geo_with_ll2.py -input {} -output {} -lat {} -lon {} -latMin {} -latMax {} -lonMin {} -lonMax {} -ssize {} -rmethod {}".format(SCR_DIR,
#            corLookedFile, 
#            corLookedGeoFile,
#            latLookedFile,
#            lonLookedFile,
#            bbox,
#            latMin,
#            latMax,
#            lonMin,
#            lonMax,
#            inps.ssize,
#            1)
#        runCmd(cmd)

        #os.remove(ifgFile)
        #os.remove(ampFile)
        #os.remove(ampLookedFile)
        #os.remove(ifgLookedFile)
        #os.remove(corLookedFile)
        #os.remove(latLookedFile)
        #os.remove(lonLookedFile)

        #os.remove(ifgFile+'.xml')
        #os.remove(ampFile+'.xml')
        #os.remove(ampLookedFile+'.xml')
        #os.remove(ifgLookedFile+'.xml')
        #os.remove(corLookedFile+'.xml')
        #os.remove(latLookedFile+'.xml')
        #os.remove(lonLookedFile+'.xml')

        #os.remove(ifgFile+'.vrt')
        #os.remove(ampFile+'.vrt')
        #os.remove(ampLookedFile+'.vrt')
        #os.remove(ifgLookedFile+'.vrt')
        #os.remove(corLookedFile+'.vrt')
        #os.remove(latLookedFile+'.vrt')
        #os.remove(lonLookedFile+'.vrt')


# USAGE
# > slcp2cor.py -mdir ${dirm} -sdir ${dirs} -gdir ${dirg} -rlks 7 -alks 3 -ssize 1.0  

