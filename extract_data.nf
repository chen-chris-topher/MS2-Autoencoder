#!/usr/bin/env nextflow
params.inputSpectra = "./spectra_data_2/*mzXML"
params.outdir = "$baseDir/output_nf_2"
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

        $TOOL_FOLDER/msconvert "$inputFile" --outfile "${file_id}.mzXML" --mzXML

        mkdir "${file_id}_outdir"
        /Users/cmaceves/miniconda3/envs/autoencoder/bin/python "$TOOL_FOLDER"/main.py "${file_id}.mzXML" "${file_id}_outdir"
        rm "${file_id}.mzXML"
		"""
    else if ( extension == 'mzXML' )
        """
		if( [ -d "${file_id}_outdir" ] )
		then
			echo "${file_id} directory made"
		else
        	mkdir "${file_id}_outdir"
        	/Users/cmaceves/miniconda3/envs/autoencoder/bin/python "$TOOL_FOLDER"/main.py "$inputFile" "${file_id}_outdir"
        	rm "${file_id}.mzXML"
		fi
        """
    else
        error "Invalid Extension"	

}
