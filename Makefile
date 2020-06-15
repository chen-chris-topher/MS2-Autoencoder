test:
	python ./bin/main.py test_data/Cholesterol_130uM_GB1_01_6914.mzXML output_dir

test_workflow:
	nextflow run extract_data.nf -resume	

test_cluster_Chemicalstandards:
	nextflow run extract_data.nf -c cluster.config --inputSpectra="/data/massive/MSV000078556/ccms_peak/Chemicalstandards/*" -with-trace -resume
	
MSV000082048:	
	nextflow run extract_data.nf -c cluster.config --inputSpectra="/data/massive/MSV000082048/peak/**.mzML" -with-trace -resume

MSV000082582:
	nextflow run extract_data.nf -c cluster.config --inputSpectra="/data/massive/MSV000082869/peak/Colgate_phase2/**.mzXML" -with-trace -resume
