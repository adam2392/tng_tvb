import os
import numpy as np 
import json
from format_data_utils import *
from util_edf import ExtractEDF

class Dataset(object):
    json_obj = {
        "filename": '',
        "onset": '',
        "termination": '',
        "bad_channels": [],
        "non_eeg_channels": [],
        "type": "",
        "note": "",
        "patientid": "", # after this point, are points added by Adam Li
        "outcome": "",
        "engel_score": "",
        "ez_elecs": [],
        "resect_elecs": [],
        "clinical_center": "",
    }
    iimarkers = ['aslp', 'aw', 'ii', 'inter']
    szmarkers = ['sz', 'seiz', 'cris', 'ictal', 'absence']
    stimmarkers = ['stim']
    def __init__(self, patientid="", filename="", onset=None, termination=None, 
                        engel_score=None, outcome=None, ez_elecs=[], resect_elecs=[],
                        bad_channels=[], non_eeg_channels=[], clinical_center=""):
        self.json_obj['filename'] = filename
        self.json_obj['patientid'] = patientid
        self.json_obj['onset'] = onset 
        self.json_obj['termination'] = termination
        self.json_obj['engel_score'] = engel_score
        self.json_obj['outcome'] = outcome
        self.json_obj['ez_elecs'] = ez_elecs
        self.json_obj['resect_elecs'] = resect_elecs
        self.json_obj['bad_channels'] = bad_channels
        self.json_obj['non_eeg_channels'] = non_eeg_channels
        self.json_obj['clinical_center'] = clinical_center

        if any(x in filename for x in self.iimarkers):
            self.json_obj['type'] = 'Interictal'
        if any(x in filename for x in self.szmarkers):
            self.json_obj['type'] = 'Seizure'
        if onset is not None:
            self.json_obj['type'] = 'Seizure'

        if any(x in filename for x in self.stimmarkers):
            self.json_obj['type'] = 'Stim'

all_patients = [
        'id001_ac', 'id002_cj', 'id003_cm', 'id004_cv',
        'id005_et', 'id006_fb', 'id007_fb', 'id008_gc',
        'id009_il', 'id010_js', 'id011_ml', 'id012_pc',
        'id013_pg', 'id014_rb'                     # 14

        # 'ummc001', 'ummc002', 'ummc003', 'ummc004',
        # 'ummc005', 'ummc006', 'ummc007', 'ummc008',
        # 'ummc009',                               # 9

        # 'pt1', 'pt2', 'pt3', 'pt6', 'pt7', 
        # 'pt4', 'pt8',
        # 'pt10', 'pt11', 'pt12', 'pt13', 'pt14', 'pt15',
        # 'pt16', 'pt17'                           # 15

        # 'jh103', 'jh105'                         # 2

        # 'la01', 'la02', 'la03', 'la04', 'la05',  # 16
        # 'la06', 'la07', 'la08', 'la09', 'la10',
        # 'la11', 'la12', 'la13', 'la15',
        # 'la16', 'la17'
]

if __name__ == '__main__':
    clinical_center = 'ummc'
    datadir = '/Volumes/ADAM LI/rawdata/pipelinedata/_{}/'.format(clinical_center)
    datadir = '/Volumes/ADAM LI/rawdata/old'
    NON_EEG = ['dc', 'ekg', 'ref', 'emg']

    for patientid in all_patients:
        print("On patient: ", patientid)
        tsdir = os.path.join(datadir, patientid)

        # get a list of all the datasets for this patient
        filepaths = []
        for root, dirs, files in os.walk(tsdir):
            for file in files:
                if '.edf' in file:
                    filepaths.append(os.path.join(root, file))
        print("Found {} edf files".format(len(filepaths)))

        # for each file create a json object
        for file in filepaths:
            try:
                print("On file: {}".format(file))
                # get patient name
                direc, filename = os.path.split(file)
                pid = ''.join(filename.lower().split('_')[0:2])
                szid = ''
                # split up sz id and patid 
                # pid, szid = splitpatient(filename.lower())

                print("looking at : ", pid, szid)
                # get hard included chans -> get the bad chans index
                included_indices, ez_elecs, resect_elecs, engel_score = returnindices(pid, seiz_id=szid)
                if engel_score == 1:
                    outcome = 'success'
                elif engel_score > 1:
                    outcome = 'failure'
                elif engel_score == -1:
                    outcome = 'na'
                ez_elecs = [chan.lower() for chan in ez_elecs]
                resect_elecs = [chan.lower() for chan in resect_elecs]

                included_indices = np.array(included_indices) + 1

                
                # now extract necessary data from edf file
                extractor = ExtractEDF(file)
                onset = extractor.onset_sec 
                termination = extractor.offset_sec
                chanlabels = extractor.chanlabels
                chanlabels = np.array([chan.lower() for chan in chanlabels])

                # print(chanlabels)
                # if onset is not None:
                #     onset = None
                # if termination is not None:
                #     termination = None

                # get the bad channels and non_eeg channels
                goodchannels = chanlabels[included_indices]
                # print("included indices: ", included_indices)
                # print("good channels: ", goodchannels)
                bad_channels = [chan.lower() for chan in chanlabels if chan not in goodchannels]
                non_eeg_channels = [chan.lower() for chan in chanlabels if any(x in chan for x in NON_EEG)]
                
                # non_eeg_channels = list(non_eeg_channels)
                # bad_channels = list(set(bad_channels) - set(non_eeg_channels))

                # create dataset object
                dataset = Dataset(patientid, filename=filename, onset=onset, termination=termination, 
                            engel_score=engel_score, outcome=outcome, ez_elecs=ez_elecs, resect_elecs=resect_elecs,
                            bad_channels=bad_channels, non_eeg_channels=non_eeg_channels, clinical_center=clinical_center)
                            
                # save the final json object
                root, ext = os.path.splitext(filename)
                metafilename = os.path.join(direc, root + '.json')

                # print(dataset.json_obj)

                # save the timepoints, included channels used, parameters
                # dumped = json.dumps(dataset.json_obj)
                with open(metafilename, 'w') as f:
                    json.dump(dataset.json_obj, f, indent=4)#, separators=(',', ':'))
                print(goodchannels)
                print(bad_channels)
                print(non_eeg_channels)
            except Exception as e:
                print(e)
                print(file, " failed")
        #     break
        # break