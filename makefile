SHELL := /usr/bin/env bash

TNG_USER=adamli
ssh                             := ssh $(port)

######################### Syncing to Cloud #########################
# download a tvb-pipeline patient data from processed directory
*download-patient:
	@{ rsync -aP adamli@139.124.148.56:/home/vep/Pipeline/1-Processed/${patient}/tvb/ /Users/adam2392/Documents/tvb/metadata/${patient}/;\
	rsync -aP adamli@139.124.148.56:/home/vep/Pipeline/1-Processed/${patient}/seeg/*.xyz /Users/adam2392/Documents/tvb/metadata/${patient}/;\
}
# rsync -aP adamli@139.124.148.56:/home/vep/Pipeline/0-Raw/id001_ac/elec/elec* /Users/adam2392/Documents/;\

# download all the eeg data from tvb pipeline (seeg)
*download-eeg:
	@{ rsync -aP adamli@139.124.148.56:/home/vep/Pipeline/0-Raw/${patient}/seeg/*.eeg /Users/adam2392/Downloads/${patient}/;\
	rsync -aP adamli@139.124.148.56:/home/vep/Pipeline/0-Raw/${patient}/seeg/*.edf /Users/adam2392/Downloads/${patient}/;\
}


# download fragility results for tvb sim
*download-pert:
	@{ rsync -aP adamli@139.124.148.56:/home/adamli/data/output/tvbsim/pert/ /Volumes/ADAM\ LI/pydata/output/tvbsim/pert/;\
}
*download-mvar:
	@{ rsync -aP adamli@139.124.148.56:/home/adamli/data/output/tvbsim/ /Volumes/ADAM\ LI/pydata/output/tvbsim/mvar/;\
}

*download-fft:
	@{ rsync -aP adamli@139.124.148.56:/home/adamli/data/output/frequencyanalysis/ /Volumes/ADAM\ LI/pydata/output/frequencyanalysis/;\
}

# *download-tvbsim-results:
# 	@{ rsync -aP adamli@139.124.148.56:/home/adamli/data/output/tvbsim/ /Volumes/ADAM\ LI/pydata/output/tvbsim/;\
# }

# download the raw tvb sim
*download-tvbsims:
	@{ rsync -aP adamli@139.124.148.56:/home/adamli/data/tvbforwardsim/ /Volumes/ADAM\ LI/pydata/tvbforwardsim/;\
}

*push-mvar:
	@{ rsync -aP /Users/adam2392/Documents/pydata/output/mvar/ adamli@139.124.148.56:/home/adamli/data/output/mvar/;\
}

# rsync -aP /Users/adam2392/Documents/tvb/metadata/ adamli@139.124.148.56:/home/adamli/metadata/;
# rsync -aP /Volumes/ADAM\ LI/pydata/output/mvar/ adamli@139.124.148.56:/home/adamli/data/;
# rsync -aP /Volumes/ADAM\ LI/pydata/converted/ adamli@139.124.148.56:/home/adamli/data/converted/;
######################### Functions to Make #########################
download: *download-patient
download-eeg: *download-eeg
download-pert: *download-pert
download-mvar: *download-mvar
download-freq: *download-fft

download-tvbsims: *download-tvbsims