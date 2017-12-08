SHELL := /usr/bin/env bash

TNG_USER=adamli
ssh                             := ssh $(port)

######################### Syncing to Cloud #########################
# Download data up to the cloud
# *sync-data:
#   @{ rsync -aP adamli@139.124.148.56:/home/vep/Pipeline/0-Raw/id005_et/seeg/*.edf /Users/adam2392/Downloads;\
#   # rsync -aP adamli@139.124.148.56:/home/vep/Pipeline/1-Processed/${patient}/seeg/*.xyz /Users/adam2392/Downloads/${patient};
#   # rsync -aP adamli@139.124.148.56:/home/vep/Pipeline/1-Processed/${patient}/seeg/gain_inv* /Users/adam2392/Downloads/${patient};
#   # rsync -aP adamli@139.124.148.56:/home/vep/Pipeline/1-Processed/${patient}/tvb/connectivity.zip /Users/adam2392/Downloads/${patient};
# }

*download-patient:
	@{ rsync -aP adamli@139.124.148.56:/home/vep/Pipeline/1-Processed/${patient}/tvb/ /Users/adam2392/Documents/tvb/metadata/${patient}/;\
}
# rsync -aP adamli@139.124.148.56:/home/vep/Pipeline/1-Processed/${patient}/tvb/surface_cort.zip /Users/adam2392/Downloads/${patient}/;\
# rsync -aP adamli@139.124.148.56:/home/vep/Pipeline/1-Processed/${patient}/seeg/*.xyz /Users/adam2392/Downloads/${patient}/;\
# rsync -aP adamli@139.124.148.56:/home/vep/Pipeline/1-Processed/${patient}/seeg/gain_inv* /Users/adam2392/Downloads/${patient}/;\
# rsync -aP adamli@139.124.148.56:/home/vep/Pipeline/1-Processed/${patient}/tvb/connectivity.zip /Users/adam2392/Downloads/${patient}/;\
# rsync -aP adamli@139.124.148.56:/home/vep/Pipeline/1-Processed/${patient}/tvb/surface_cort.zip /Users/adam2392/Downloads/${patient}/;\
# }

*download-eeg:
	@{ rsync -aP adamli@139.124.148.56:/home/vep/Pipeline/0-Raw/${patient}/seeg/*.eeg /Users/adam2392/Downloads/${patient}/;\
	rsync -aP adamli@139.124.148.56:/home/vep/Pipeline/0-Raw/${patient}/seeg/*.edf /Users/adam2392/Downloads/${patient}/;\
}

rsync -aP adamli@139.124.148.56:/home/adamli/data/tvbforwardsim/ /Users/adam2392/Documents/pydata/tvbforwardsim/ ;

# rsync -aP /Users/adam2392/Documents/pydata/output/mvar/ adamli@139.124.148.56:/home/adamli/data/;
# rsync -aP /Users/adam2392/Documents/tvb/metadata/ adamli@139.124.148.56:/home/adamli/metadata/;
# rsync -aP /Volumes/ADAM\ LI/pydata/output/mvar/ adamli@139.124.148.56:/home/adamli/data/;
# rsync -aP /Volumes/ADAM\ LI/pydata/converted/ adamli@139.124.148.56:/home/adamli/data/converted/;
######################### Functions to Make #########################
# sync-to-remote-data: *sync-data
download: *download-patient
downloadraw: *download-eeg
