#!/usr/bin/env nextflow
params.inputSpectra = "./data/*mzXML"
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
    
    if( extension == 'mzML' )
        """
        export LC_ALL=C

        $TOOL_FOLDER/msconvert "$inputFile" --outfile "${file_id}.mzXML" --mzXML

        mkdir "${file_id}_outdir"
        python $TOOL_FOLDER/main.py "${file_id}.mzXML" "${file_id}_outdir"
        """
    else if ( extension == 'mzXML' )
        """
        mkdir "${file_id}_outdir"
        python $TOOL_FOLDER/main.py "$inputFile" "${file_id}_outdir"
        rm "${file_id}.mzXML"
        """
    else
        error "Invalid Extension"

}
