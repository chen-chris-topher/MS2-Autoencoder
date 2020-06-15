test:
	python ./bin/main.py test_data/Cholesterol_130uM_GB1_01_6914.mzXML output_dir

test_workflow:
	nextflow run extract_data.nf -resume	

test_cluster_Chemicalstandards:
	nextflow run extract_data.nf -c cluster.config --inputSpectra="/data/massive/MSV000078556/ccms_peak/Chemicalstandards/*" -with-trace -resume
	
MSV000082048:	
	nextflow run extract_data.nf -c cluster.config --inputSpectra="/data/massive/MSV000082048/peak/Plate_22/**.mzML" -with-trace -resume
	nextflow run extract_data.nf -c cluster.config --inputSpectra="/data/massive/MSV000082048/peak/Plate_23/**.mzML" -with-trace -resume
	nextflow run extract_data.nf -c cluster.config --inputSpectra="/data/massive/MSV000082048/peak/Plate_24/**.mzML" -with-trace -resume
	nextflow run extract_data.nf -c cluster.config --inputSpectra="/data/massive/MSV000082048/peak/Plate_25/**.mzML" -with-trace -resume
	nextflow run extract_data.nf -c cluster.config --inputSpectra="/data/massive/MSV000082048/peak/Plate_26/**.mzML" -with-trace -resume
	nextflow run extract_data.nf -c cluster.config --inputSpectra="/data/massive/MSV000082048/peak/Plate_27/**.mzML" -with-trace -resume
	nextflow run extract_data.nf -c cluster.config --inputSpectra="/data/massive/MSV000082048/peak/Plate_28/**.mzML" -with-trace -resume
MSV000082582:
	nextflow run extract_data.nf -c cluster.config --inputSpectra="/data/massive/MSV000082869/peak/Colgate_phase2/Plate1/**.mzXML" -with-trace -resume
	nextflow run extract_data.nf -c cluster.config --inputSpectra="/data/massive/MSV000082869/peak/Colgate_phase2/Plate10/**.mzXML" -with-trace -resume
	nextflow run extract_data.nf -c cluster.config --inputSpectra="/data/massive/MSV000082869/peak/Colgate_phase2/Plate1_stained/**.mzXML" -with-trace -resume
	nextflow run extract_data.nf -c cluster.config --inputSpectra="/data/massive/MSV000082869/peak/Colgate_phase2/Plate2/**.mzXML" -with-trace -resume
	nextflow run extract_data.nf -c cluster.config --inputSpectra="/data/massive/MSV000082869/peak/Colgate_phase2/Plate2_stained/**.mzXML" -with-trace -resume
	nextflow run extract_data.nf -c cluster.config --inputSpectra="/data/massive/MSV000082869/peak/Colgate_phase2/Plate3/**.mzXML" -with-trace -resume
	nextflow run extract_data.nf -c cluster.config --inputSpectra="/data/massive/MSV000082869/peak/Colgate_phase2/Plate4/**.mzXML" -with-trace -resume
	nextflow run extract_data.nf -c cluster.config --inputSpectra="/data/massive/MSV000082869/peak/Colgate_phase2/Plate5/**.mzXML" -with-trace -resume
	nextflow run extract_data.nf -c cluster.config --inputSpectra="/data/massive/MSV000082869/peak/Colgate_phase2/Plate6/**.mzXML" -with-trace -resume
	nextflow run extract_data.nf -c cluster.config --inputSpectra="/data/massive/MSV000082869/peak/Colgate_phase2/Plate7/**.mzXML" -with-trace -resume
	nextflow run extract_data.nf -c cluster.config --inputSpectra="/data/massive/MSV000082869/peak/Colgate_phase2/Plate8/**.mzXML" -with-trace -resume
	nextflow run extract_data.nf -c cluster.config --inputSpectra="/data/massive/MSV000082869/peak/Colgate_phase2/Plate9/**.mzXML" -with-trace -resume
	nextflow run extract_data.nf -c cluster.config --inputSpectra="/data/massive/MSV000082869/peak/Colgate_phase2/Plate1/**.mzXML" -with-trace -resume
