#!/usr/bin/env nextflow
params.inputSpectra = "./spectra_data_1/*{.mzXML,.mzML}"
params.outdir = "$baseDir/output_nf_1"
TOOL_FOLDER = "$baseDir/bin"

process extractPairs { 
    errorStrategy 'ignore'
    //errorStrategy 'terminate'
    echo true
    //validExitStatus 1
    publishDir "$params.outdir", mode: 'copy'

    input:
    set file_id, extension, file(inputFile) from Channel.fromPath( params.inputSpectra ).map { file -> tuple(file.baseName, file.extension, file) }

    output:
    file "*_outdir"

    script:
    println(extension)
    if( extension == 'mzML' )
        """
        	export LC_ALL=C

        	"$TOOL_FOLDER"/msconvert "$inputFile" --outfile "${file_id}.mzXML" --mzXML
        	mkdir "${file_id}_outdir"
        	echo "${file_id}"
		/Users/cmaceves/miniconda3/envs/autoencoder/bin/python "$TOOL_FOLDER"/main_optimized.py "${file_id}.mzXML" "${file_id}_outdir"
		
	"""
    else if ( extension == 'mzXML' )
        """
	       	mkdir "${file_id}_outdir"
        	/Users/cmaceves/miniconda3/envs/autoencoder/bin/python "$TOOL_FOLDER"/main_optimized.py "$inputFile" "${file_id}_outdir"
  	
        """
    else
        error "Invalid Extension"	

}
