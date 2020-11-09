#!/usr/bin/env python3

#Cunren Liang, JPL/Caltech


import os
import sys
import glob
import shutil
import ntpath
import pickle
import argparse
import datetime
import numpy as np
from scipy.interpolate import griddata
from scipy.signal import convolve2d as conv2

import isce
import isceobj
from isceobj.Image import createImage,createDemImage


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


def cmdLineParse():
    '''
    Command line parser.
    '''
    parser = argparse.ArgumentParser( description='geocode file using lat and lon files',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="")
    parser.add_argument('-input', dest='input', type=str, required=True,
            help = 'input file to be geocoded')
    parser.add_argument('-output', dest='output', type=str, required=True,
            help = 'output file to be geocoded')
    parser.add_argument('-lat', dest='lat', type=str, required=True,
            help = 'latitude file.')
    parser.add_argument('-lon', dest='lon', type=str, required=True,
            help = 'longitude file')
    parser.add_argument('-bbox', dest='bbox', type=str, required=False,
            help='geocode bounding box (format: S/N/W/E). or you can input master frame pickle file that contains a bounding box')
    parser.add_argument('-latMin', dest='latMin', type=float, required=False,default=0,
            help='geocode bounding box (format: S/N/W/E). or you can input master frame pickle file that contains a bounding box')
    parser.add_argument('-latMax', dest='latMax', type=float, required=False,default=0,
            help='geocode bounding box (format: S/N/W/E). or you can input master frame pickle file that contains a bounding box')
    parser.add_argument('-lonMin', dest='lonMin', type=float, required=False,default=0,
            help='geocode bounding box (format: S/N/W/E). or you can input master frame pickle file that contains a bounding box')
    parser.add_argument('-lonMax', dest='lonMax', type=float, required=False,default=0,
            help='geocode bounding box (format: S/N/W/E). or you can input master frame pickle file that contains a bounding box')
    parser.add_argument('-ssize', dest='ssize', type=float, default=1.0,
            help = 'output sample size. default: 1.0 arcsec')
    parser.add_argument('-rmethod', dest='rmethod', type=int, default=1,
            help = 'resampling method. 0: nearest. 1: linear (default). 2: cubic.')

    return parser.parse_args()


if __name__ == '__main__':

    inps = cmdLineParse()

    inImage = createImage()
    inImage.load(inps.input+'.xml')
    inImage.filename = inps.input

    width = inImage.width
    length = inImage.length
    scheme = inImage.scheme
    nbands = inImage.bands
    print("number of bands : {}".format(nbands))
    datatype = inImage.dataType

    #check file width and length
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
     
    if inps.latMin!=0 and inps.latMax!=0 and inps.lonMin!=0 and inps.lonMax!=0:
         bbox = ( inps.latMin, inps.latMax, inps.lonMin, inps.lonMax )
    else: 
         bbox = (0,0,0,0)
#    bbox[1] = inps.latMax
 #   bbox[2] = inps.lonMin
 #   bbox[3] = inps.lonMax


    print("geocode bounding box:")
    print("south: {}".format(bbox[0]))
    print("north: {}".format(bbox[1]))
    print("west: {}".format(bbox[2]))
    print("east: {}".format(bbox[3]))

    #get input bands saved in a list
    print("reading data to be geocoded")
    print("{}   {}   {}".format(width,length,datatype))
    bands = read_bands(inps.input, length, width, scheme, nbands, datatype)

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
    print("{}".format(np.min(lat2)))
    #get longitude
    print("reading longitude")
    lon = read_bands(inps.lon, length, width, lonImage.scheme, lonImage.bands, lonImage.dataType)
    lon = lon[0]
    lon2=lon
    lon2[lon2==0]=np.nan
    print("{}".format(np.min(lon2)))
#    if inps.latMin==0 and inps.latMax==0 and inps.lonMin==0 and inps.lonMax==0:
#         bbox = ( np.nanmin(lat2), np.nanmax(lat2), np.nanmin(lon2), np.nanmax(lon2) )

#    print("geocode bounding box:")
#    print("south: {}".format(bbox[0]))
#    print("north: {}".format(bbox[1]))
#    print("west: {}".format(bbox[2]))
#    print("east: {}".format(bbox[3]))


    latlon = np.zeros((length*width, 2), dtype=np.float64)
    lat=lat.reshape(length*width)
    lon=lon.reshape(length*width)
    diffll=np.abs(lat2-lat2m)
    diffll[np.isnan(diffll)]=0
    lat2[lat2==0]=np.nan   
    mask= (diffll>0.1) | (np.isnan(lat2)) | ( np.isnan(lat2m)) | ( np.isnan(bands[0].reshape(length*width)))
    lat=lat[mask==0] 
    lon=lon[mask==0] 
    if inps.latMin==0 and inps.latMax==0 and inps.lonMin==0 and inps.lonMax==0:
         bbox = ( np.nanmin(lat), np.nanmax(lat), np.nanmin(lon), np.nanmax(lon) )
    print("geocode bounding box:")
    print("south: {}".format(bbox[0]))
    print("north: {}".format(bbox[1]))
    print("west: {}".format(bbox[2]))
    print("east: {}".format(bbox[3]))

    if inps.latMin!=0 and inps.latMax!=0 and inps.lonMin!=0 and inps.lonMax!=0:

        submask= (lat>bbox[0]) & (lat<bbox[1]) & (lon<bbox[3]) & (lon>bbox[2])
        
        lat=lat[submask]
        lon=lon[submask]
        print("cropping image {}-> {}".format(len(mask),len(lat)))
        



    latlon = np.zeros( (len(lat), 2), dtype=np.float64)
    latlon[:, 0] = lat
    latlon[:, 1] = lon
    bands2=np.zeros( (nbands, len(lat)), dtype=np.float32)
    for i in range(nbands):
       temp=bands[i].reshape(length*width)
       temp=temp[mask==0]
       if inps.latMin!=0 and inps.latMax!=0 and inps.lonMin!=0 and inps.lonMax!=0:
           temp=temp[submask]
       bands2[i]=temp
    #                             n       s                         w       e
    grid_lat, grid_lon = np.mgrid[bbox[1]:bbox[0]:-sample_size_lat, bbox[2]:bbox[3]:sample_size_lon]
    print("geocoded image dimension {} {}".format(np.shape(grid_lat)[0],np.shape(grid_lat)[1]))
    msk=np.zeros([length,width])
    msk[0:2,:]=1
    msk[-2:,:]=1
    msk[:,-2:]=1
    msk[:,0:2]=1
    msk= msk.reshape(length*width)
    msk=msk[mask==0]
    if inps.latMin!=0 and inps.latMax!=0 and inps.lonMin!=0 and inps.lonMax!=0:
       msk=msk[submask]
    geomsk = griddata(latlon, msk, (grid_lat, grid_lon), method=rmethod, fill_value = 0.0)
    print("interpolate input data")
    bands_geo = []
    for i in range(nbands):
        geoband = griddata(latlon, (bands2[i]), (grid_lat, grid_lon), method=rmethod, fill_value = 0.0)
        #geoband = griddata(latlon, (bands[i]).reshape(length*width), (grid_lat, grid_lon), method=rmethod, fill_value = 0.0)
#        bands_geo.append(geoband)
        bands_geo.append(np.multiply(geoband,(geomsk==0) ))

    print("write result")
    (length_geo, width_geo) = geoband.shape
    write_bands(inps.output, length_geo, width_geo, scheme, nbands, datatype, bands_geo)


    outImage = inImage
    outImage.setFilename(inps.output)
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



#./geo.py -input filt_diff_150909-160824_8rlks_16alks_msk.unw -output filt_diff_150909-160824_8rlks_16alks_msk.unw.geo -lat 150909-160824_8rlks_16alks.lat -lon 150909-160824_8rlks_16alks.lon -bbox 150909.slc.pck -ssize 1.0


