import os
import argparse 

# import frequency analysis runners
from run_freq import FreqAnalysis
from fragility.io.loadsimdataset import LoadSimDataset

FREQ_MODES = ['fft', 'morlet']
DATA_TYPE = ['sim']

def run_freq(metadata, rawdata, mode, outputfilename, outputmetafilename):
    if not mode in FREQ_MODES:
        raise Exception("mode should be either 'fft', or 'morlet'. Not {}".format(mode))

    if mode == 'fft':
        # FFT Parameters
        mtbandwidth = 4
        mtfreqs = []
        fftargs = {"winsize": metadata['winsize'], 
                    "stepsize": metadata['stepsize'],
                    "samplerate": metadata['samplerate'], 
                    "mtfreqs":mtfreqs, 
                    "mtbandwidth":mtbandwidth}
        # run fft
        power, freqs, phase, timepoints = FreqAnalysis.run_fft(rawdata, fftargs)

        # add to metadata
        metadata['mtbandwidth'] = mtbandwidth
        metadata['freqs'] = freqs
    elif mode == 'morlet':
        # Wavelet Parameters
        waveletfreqs = 2**(np.arange(1.,9.,1./5))
        waveletwidth = 6
        waveletargs = {"winsize": metadata['winsize'], 
                        "stepsize": metadata['stepsize'],
                        "samplerate": metadata['samplerate'], 
                        "waveletfreqs":waveletfreqs, 
                        "waveletwidth":waveletwidth}
        # run morlet
        power, phase, timepoints = FreqAnalysis.run_morlet(rawdata, waveletargs)

        metadata['waveletfreqs'] = waveletfreqs
        metadata['waveletwidth'] = waveletwidth

    # add the consistent parameters
    metadata['winsize'] = winsize
    metadata['stepsize'] = stepsize
    metadata['timepoints'] = timepoints
    metadata['rawfilename'] = datafile

    # save the data
    run_freq.save_data(outputfilename, outputmetafilename, power, phase, metadata)

def load_raw_data(patdatadir, datafile, metadatadir, patient, reference):
    loader = LoadSimDataset(root_dir=patdatadir, 
                                datafile=datafile, 
                                rawdatadir=metadatadir, 
                                patient=patient,
                                reference=reference, 
                                preload=True)
    # get filtered/referenced rawdata
    rawdata = loader.rawdata 
    metadata = loader.getmetadata()

    return rawdata, metadata

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('patient', help="Patient to analyze")
    parser.add_argument('analysis_type', help="Type of analysis ('mvar', 'pert', 'fft', 'morlet'")
    parser.add_argument('--winsize', default='500', type=int, help="Window size to use in model.")
    parser.add_argument('--stepsize', default='250', type=int, help="step size to use in model.")
    parser.add_argument('--radius', default='', help="Radius of perturbation to use in model.")
    parser.add_argument('--rawdatadir', help="TVB Simulated data directory, or raw data directory to find the data.")
    parser.add_argument('--tempdatadir',  help="Where to save the temporary computations per window.")
    parser.add_argument('--outputdatadir',  help="Where to save the output data.")
    parser.add_argument('--metadatadir',  help="Where the metadata for the TVB sims is.")
    parser.add_argument('--datafile', default='',  help="The datafile to analyze ")
    parser.add_argument('--idatafile', default=0, type=int,  help="The datafile to analyze index (e.g. 1-9)")
    parser.add_argument('--taskid', type=int,  help="The task id used here.")
    parser.add_argument('--numtasks', type=int,  help="The total number of tasks assigned to this job.")
    parser.add_argument('--fixwins', type=int, default=1, help="Should we fixjobs?")
    parser.add_argument('--perturbtype', default='C',  help="Perturbation type (R or C).")
    parser.add_argument('--datatype', help="Data type: either 'sim', or 'real'")
    parser.add_argument('--RUNMERGE', default=0, type=bool, help="Should we run merging?")


    args = parser.parse_args()

    # extract passed in variable
    # patient = args.patient
    # analysis_type = args.analysis_type
    # winsize = args.winsize
    # stepsize = args.stepsize
    # radius = float(args.radius)

    # rawsimdatadir = args.rawdatadir
    # tempdatadir = args.tempdatadir
    # outputdatadir = args.outputdatadir
    # metadatadir = args.metadatadir

    # idatafile = args.idatafile-1
    # datafile = args.datafile
    # datatype = args.datatype
    # taskid = args.taskid-1
    # numtasks = args.numtasks

    # elif analysis_type == 'fft' or analysis_type == 'morlet':
    #     # get datafile
    #     rawdata, metadata = load_raw_data(patdatadir, datafile, metadatadir, patient, reference, mode=datatype)

    #     # create checker for num wins
    #     outputdir = os.path.join(outputdatadir, 'freq', analysis_type, patient)
    #     if not os.path.exists(outputdir):
    #         os.makedirs(outputdir)

    #     # where to save final computation
    #     outputfilename = os.path.join(outputdir, 
    #             '{}_{}_{}model.npz'.format(patient, analysis_type, idatafile))
    #     outputmetafilename = os.path.join(outputdir,
    #         '{}_{}_{}meta.json'.format(patient, analysis_type, idatafile))

    #     ###################### 1. LOAD IN DATA ##############################
    #     rawdata, metadata = load_realdata(patdatadir, datafile, metadatadir, patient) 
    #     run_freq(metadata, rawdata, analysis_type, outputfilename, outputmetafilename)
    #         