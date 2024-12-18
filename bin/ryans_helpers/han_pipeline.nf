#!/usr/bin/env nextflow

/*
 * HAN (Hierarchical Attention Network) Pipeline for Genomic Sequence Analysis
 * 
 * This pipeline processes genomic sequences to predict regulatory activity in the centermost
 * 128bp window using the entire sequence context (131,072bp). The model uses a hierarchical
 * structure with CNN-based motif detection and attention mechanisms at multiple levels.
 *
 * Input:
 * - Basenji-formatted TFRecord files containing sequences and their regulatory activity values
 *
 * Output:
 * - Trained HAN model
 * - Performance metrics and visualizations
 * - Evaluation results on test set
 */

// Pipeline parameter defaults
params.output_base = "/clusterfs/nilah/ryank/proj/compartments/20241218_HAN_basenji"
params.data_dir = "${params.output_base}/data/processed_han"
params.model_dir = "${params.output_base}/models/han"
params.results_dir = "${params.output_base}/results/han"
params.batch_size = 32
params.num_epochs = 100
params.learning_rate = 0.001
params.num_motif_filters = 128
params.kmer_size = 6
params.phrase_gru_size = 64
params.sent_gru_size = 64
params.num_targets = 1

// Create channel for input data
Channel
    .fromPath("${params.data_dir}")
    .ifEmpty { error "No input data directory found at: ${params.data_dir}" }
    .set { input_data }

/*
 * Process 1: Train the HAN model
 * This process runs the training script and monitors for completion
 */
process train_model {
    publishDir "${params.model_dir}", mode: 'copy'
    
    input:
    path data_dir from input_data
    
    output:
    path "checkpoints/*" into model_checkpoints
    path "TRAINING_COMPLETED" into training_completed
    path "metrics.jsonl" into training_metrics
    
    script:
    """
    python ${baseDir}/train_han.py \
        --data-dir ${data_dir} \
        --output-dir . \
        --batch-size ${params.batch_size} \
        --num-epochs ${params.num_epochs} \
        --learning-rate ${params.learning_rate} \
        --num-motif-filters ${params.num_motif_filters} \
        --kmer-size ${params.kmer_size} \
        --phrase-gru-size ${params.phrase_gru_size} \
        --sent-gru-size ${params.sent_gru_size} \
        --num-targets ${params.num_targets}
    """
}

/*
 * Process 2: Generate Performance Plots
 * Creates visualizations of training and validation metrics
 */
process generate_plots {
    publishDir "${params.results_dir}/plots", mode: 'copy'
    
    input:
    path metrics from training_metrics
    
    output:
    path "*.png" into performance_plots
    
    script:
    """
    python ${baseDir}/plot_performance.py \
        --metrics-file ${metrics} \
        --output-dir .
    """
}

/*
 * Process 3: Evaluate Model
 * Runs evaluation on the test set once training is complete
 */
process evaluate_model {
    publishDir "${params.results_dir}", mode: 'copy'
    
    input:
    path model_dir from training_completed.map { it.parent }
    path data_dir from input_data
    
    output:
    path "evaluation_results.json" into eval_results
    path "detailed_results.json" into detailed_results
    
    script:
    """
    python ${baseDir}/evaluate_han.py \
        --model-dir ${model_dir} \
        --data-dir ${data_dir} \
        --output-dir . \
        --batch-size ${params.batch_size} \
        --num-motif-filters ${params.num_motif_filters} \
        --kmer-size ${params.kmer_size} \
        --phrase-gru-size ${params.phrase_gru_size} \
        --sent-gru-size ${params.sent_gru_size} \
        --num-targets ${params.num_targets}
    """
}

/*
 * Process 4: Generate Final Report
 * Combines all results into a comprehensive report
 */
process generate_report {
    publishDir "${params.results_dir}", mode: 'copy'
    
    input:
    path eval_results
    path plots from performance_plots.collect()
    
    output:
    path "report.html" into final_report
    
    script:
    """
    python ${baseDir}/generate_report.py \
        --eval-results ${eval_results} \
        --plots-dir . \
        --output-file report.html
    """
}

workflow {
    main:
        train_model()
        generate_plots(train_model.out[2])  // metrics.jsonl
        evaluate_model(train_model.out[1].map { it.parent }, input_data)
        generate_report(evaluate_model.out[0], generate_plots.out.collect())
} 