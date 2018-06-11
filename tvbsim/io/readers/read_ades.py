import numpy as np
import os
import pandas as pd
'''
Takes in a pair file of .ades and .dat and extracts the channel names and the corresponding SEEG time series

places them into four different files

- raw numpy
- headers csv
- annotations csv
- channels csv

Which follows format that we place data from .edf files. Most data is empty since .ades does not get alot of these
data points.
'''


def rawtonumpy(raweeg, outputdatafile):
    # open output Numpy file to write
    npfile = open(outputdatafile, 'wb')
    np.save(npfile, raweeg)


def chantocsv(chanlabels, samplerate, numsamps, outputchanfile):
    ##################### 2. Import channel headers ########################
    # create list with dataframe column channel header names
    channelheaders = [[
        'Channel Number',
        'Labels',
        'Physical Maximum',
        'Physical Minimum',
        'Digital Maximum',
        'Digital Minimum',
        'Sample Frequency',
        'Num Samples',
        'Physical Dimensions',
    ]]

    # get the channel labels of file and convert to list of strings
    # -> also gets rid of excessive characters
    chanlabels = [str(x).replace('POL', '').replace(' ', '')
                  for x in chanlabels]

    # read chan header data from each chan for each column and append to list
    for i in range(len(chanlabels)):
        channelheaders.append([
            i + 1,
            chanlabels[i],
            '',
            '',
            '',
            '',
            samplerate,
            numsamps,
            '',
        ])

    # create CSV file of channel header names and data
    channelheaders_df = pd.DataFrame(data=channelheaders)
    # create CSV file of file header names and data
    channelheaders_df.to_csv(outputchanfile, index=False, header=False)


def annotationtocsv(outputannotationsfile):
    ##################### 3. Import File Annotations ########################
    # create list
    annotationheaders = [[
        'Time (sec)',
        'Duration',
        'Description'
    ]]

    for n in np.arange(0):
        annotationheaders.append([
            '',
            '',
            ''
        ])

    annotationheaders_df = pd.DataFrame(data=annotationheaders)
    # create CSV file of channel header names and data
    annotationheaders_df.to_csv(
        outputannotationsfile,
        index=False,
        header=False)


def headerstocsv(samplerate, numsamps, outputheadersfile):
    # create list with dataframe column file header names
    fileheaders = [[
        'pyedfib Version',
        'Birth Date',
        'Gender',
        'Start Date (D-M-Y)',
        'Start Time (H-M-S)',
        'Patient Code',
        'Equipment',
        'Data Record Duration (s)',
        'Number of Data Records in File',
        'Number of Annotations in File',
        'Sample Frequency',
        'Samples in File',
        'Physical Dimension'
    ]]

    # append file header data for each dataframe column to list
    # startdate = str(edffile.getStartdatetime().day) + '-' + str(edffile.getStartdatetime().month) + '-' + str(edffile.getStartdatetime().year)
    # starttime = str(edffile.getStartdatetime().hour) + '-' + str(edffile.getStartdatetime().minute) + '-' + str(edffile.getStartdatetime().second)

    fileheaders.append([
        '',
        '',
        '',
        '',
        '',
        '',
        '',
        numsamps / float(samplerate),
        '',
        '',
        samplerate,
        numsamps,
        '',
    ])
    # create dataframes from array of meta data
    fileheaders_df = pd.DataFrame(data=fileheaders)
    fileheaders_df.to_csv(outputheadersfile, index=False, header=False)


def read_ades(fname):
    dat_fname = fname.split('.ades')[0] + '.dat'
    srate = None
    nsamp = None
    sensors = []
    with open(fname, 'r') as fd:
        for line in fd.readlines():
            if line.startswith('#'):
                continue
            # parts = line.strip().split(' ')
            # lhs = parts[0]
            # rhs = parts[2]

            try:
                lhs, _, rhs = line.strip().split(' ')
            except ValueError:
                lhs = line.strip().split(' ')

            if lhs == 'samplingRate':
                srate = float(rhs)
            elif lhs == 'numberOfSamples':
                nsamp = float(rhs)
            elif lhs in ('date', 'time'):
                pass
            else:
                if isinstance(lhs, list):
                    lhs = lhs[0]
                sensors.append(lhs)
    assert srate and nsamp
    dat = np.fromfile(
        dat_fname, dtype=np.float32).reshape(
        (-1, len(sensors))).T
    return srate, sensors, dat, nsamp


if __name__ == '__main__':
    filename = './id004_cv_sz1.ades'

    # set all directories
    root_dir = os.path.join('/Users/adam2392/Downloads/')
    outputdir = '/Users/adam2392/Downloads/converted'
    # root_dir = os.path.join('/Volumes/ADAM LI/pydata/tvbforwardsim/')

    patient = 'id004_cv'
    patient = 'id015_sf'
    # patient = 'id001_ac'

    datadir = os.path.join(root_dir, patient)
    # Get ALL datafiles from all downstream files
    datafiles = []
    for root, dirs, files in os.walk(datadir):
        for file in files:
            if '.DS' not in file and '.ades' in file:
                datafiles.append(os.path.join(root, file))
    print(len(datafiles))

    def patdir(idx): return patient.lower() + '_sz' + str(idx)

    for idx, filename in enumerate(datafiles):
        print(idx, filename)

        srate, sensors, dat, nsamp = read_ades(filename)

        if not os.path.exists(os.path.join(outputdir, patdir(idx))):
            os.makedirs(os.path.join(outputdir, patdir(idx)))

        # 1. setting filename
        npyfile = os.path.join(
            outputdir,
            patdir(idx),
            patdir(idx) +
            '_rawnpy.npy')
        chanfile = os.path.join(
            outputdir,
            patdir(idx),
            patdir(idx) +
            '_chans.csv')
        headerfile = os.path.join(
            outputdir,
            patdir(idx),
            patdir(idx) +
            '_headers.csv')
        annotationsfile = os.path.join(
            outputdir,
            patdir(idx),
            patdir(idx) +
            '_annotations.csv')

        rawtonumpy(dat, npyfile)
        chantocsv(sensors, srate, nsamp, chanfile)
        headerstocsv(srate, nsamp, headerfile)
        annotationtocsv(annotationsfile)
