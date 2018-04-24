SHELL := /usr/bin/env bash

TNG_USER=adamli
ssh                             := ssh $(port)

######################### Syncing to Cloud #########################
# download a tvb-pipeline patient data from processed directory
*download-patient:
	@{ rsync -aP adamli@cluster.thevirtualbrain.org:/home/vep/Pipeline/1-Processed/${patient}/tvb/ /Users/adam2392/Documents/tvb/metadata/${patient}/;\
	rsync -aP adamli@cluster.thevirtualbrain.org:/home/vep/Pipeline/1-Processed/${patient}/seeg/*.xyz /Users/adam2392/Documents/tvb/metadata/${patient}/;\
}
# rsync -aP adamli@cluster.thevirtualbrain.org:/home/vep/Pipeline/0-Raw/id001_ac/elec/elec* /Users/adam2392/Documents/;\

# rsync -aP adamli@cluster.thevirtualbrain.org:/home/vep/data_from_tim/seeg/ ~/Downloads/tngrawdata/;\
# download all the eeg data from tvb pipeline (seeg)
*download-eeg:
	@{ rsync -aP adamli@cluster.thevirtualbrain.org:/home/vep/Pipeline/0-Raw/${patient}/seeg/*.eeg /Users/adam2392/Downloads/${patient}/;\
	rsync -aP adamli@cluster.thevirtualbrain.org:/home/vep/Pipeline/0-Raw/${patient}/seeg/*.edf /Users/adam2392/Downloads/${patient}/;\
}


# download fragility results for tvb sim
# *download-pert:
# 	@{ rsync -aP adamli@cluster.thevirtualbrain.org:/home/adamli/data/output/tvbsim/pert/ /Volumes/ADAM\ LI/pydata/output/tvbsim/*/pert/;\
# }
*download-results:
	@{ rsync -aP -z --progress adamli@cluster.thevirtualbrain.org:/home/adamli/data/output/tvbsim/ ~/Downloads/tngcluster/;\
}
# rsync -aP /Volumes/ADAM\ LI/pydata/output/tvbsim/ adamli@cluster.thevirtualbrain.org:/home/adamli/data/output/tvbsim/ 
# rsync -aPz adamli@cluster.thevirtualbrain.org:/home/adamli/data/output/tvbsim/exp009 ~/Downloads/tngcluster/

*download-fft:
	@{ rsync -aP adamli@cluster.thevirtualbrain.org:/home/adamli/data/outputfreq ~/Downloads/tngcluster/;\
}

# *download-tvbsim-results:
# 	@{ rsync -aP adamli@cluster.thevirtualbrain.org:/home/adamli/data/output/tvbsim/ /Volumes/ADAM\ LI/pydata/output/tvbsim/;\
# }

# download the raw tvb sim
*download-tvbsims:
	@{ rsync -aPz --progress adamli@cluster.thevirtualbrain.org:/home/adamli/data/tvbforwardsim/exp009 ~/Downloads/tngcluster/tvbforwardsim/;\
}
# rsync -aPz --progress adamli@cluster.thevirtualbrain.org:/home/adamli/data/tvbforwardsim/traindata/ ~/Downloads/tngcluster/tvbforwardsim/;\

# rsync -aPz adamli@cluster.thevirtualbrain.org:/home/adamli/data/output/tvbsim/exp011/seegpert/ ~/Downloads/tngcluster/;\


*push-mvar:
	@{ rsync -aP /Users/adam2392/Documents/pydata/output/mvar/ adamli@cluster.thevirtualbrain.org:/home/adamli/data/output/mvar/;\
}

*push-data:
	@{ rsync -aP /Users/adam2392/Downloads/realtng adamli@cluster.thevirtualbrain.org:/home/adamli/data/dnn/traindata_fft/;\
}

*push-fragility:
	@{ rsync -aP /Volumes/ADAM\ LI/pydata/output/ adamli@cluster.thevirtualbrain.org:/home/adamli/data/output/;\
}

# rsync -aP /Users/adam2392/Documents/tvb/_tvblibrary/tvb/simulator/models/epileptor.py adamli@cluster.thevirtualbrain.org:/home/adamli/tng_tvb/_tvblibrary/tvb/simulator/models/
# rsync -aP /Users/adam2392/Documents/tvb/metadata/ adamli@cluster.thevirtualbrain.org:/home/adamli/metadata/;
# rsync -aP /Volumes/ADAM\ LI/pydata/output/mvar/ adamli@cluster.thevirtualbrain.org:/home/adamli/data/;
# rsync -aP /Volumes/ADAM\ LI/pydata/converted/ adamli@cluster.thevirtualbrain.org:/home/adamli/data/converted/;
######################### Functions to Make #########################
download: *download-patient
download-eeg: *download-eeg
# download-pert: *download-pert
download-results: *download-results
download-freq: *download-fft

download-tvbsims: *download-tvbsims