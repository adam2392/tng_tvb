#!/bin/bash -l

patients=('id002_cj
  			id004_cv  id006_fb  id008_gc  
			id010_js  id012_pc  id014_rb  id016_hh  id018_ol 
			 id020_ct  id022_jgi  id024_ml  id026_lz 
			id001_ac    id003_cm  id005_et  id007_fb  
			id009_il  id011_ml  id013_pg  id015_sf  id017_mm
			id019_rg  id021_cf  id023_md   id025_pb
			id026_lz id027_kl id028_ag')


for patient in $patients; do
	echo $patient
	make download patient=$patient
done