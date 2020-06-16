#!/usr/bin/env nextflow
params.inputSpectra = "./test_data/*mzXML"
params.outdir = "$baseDir/output_nf_2"

TOOL_FOLDER = "$baseDir/bin"

// Extracting Pairs in Parallel
process extractPairs { 
    errorStrategy 'ignore'
    //errorStrategy 'terminate'
    //echo true
    //validExitStatus 1

    publishDir "$params.outdir/extracted_data", mode: 'copy'

    input:
    set file_id, extension, file(inputFile) from Channel.fromPath( params.inputSpectra ).map { file -> tuple(file.baseName, file.extension, file) }
 
    output:
    file "*_outdir" into extracted_folder_ch

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

// Merging Data Together
process condensePairs {
    echo true
    cache false

    memory '60 GB'

    publishDir "$params.outdir/stitched_data", mode: 'copy'

    input:
    file "input_dir/*" from extracted_folder_ch.collect()

    output:
    file "concat.hdf5"

    script:
    """
    python $TOOL_FOLDER/processing.py input_dir ready_array2.npz --name concat.hdf5
    """
}