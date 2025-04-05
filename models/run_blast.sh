#!/bin/bash

input_dir="pH_classes"
output_dir="blast_results"
mkdir -p "$output_dir"

db_path="uniprot_db/uniprot_db"

# Calculate total sequences correctly
total_sequences=$(grep -hc "^>" "$input_dir"/*.fasta | awk '{s+=$1} END {print s}')
current_sequence=0

show_progress() {
    local width=50
    local percent=$1
    local completed=$((width * percent / 100))
    local remaining=$((width - completed))
    printf "\r[%-${width}s] %d%%" "$(printf '#%.0s' $(seq 1 $completed))" "$percent"
}

for fasta_file in "$input_dir"/*.fasta; do
    filename=$(basename "$fasta_file" .fasta)
    
    class_dir="$output_dir/$filename"
    mkdir -p "$class_dir"
    
    echo "Processing file: $fasta_file"
    
    while read -r line; do
        if [[ ${line:0:1} == ">" ]]; then
            seq_id=$(echo "$line" | cut -d' ' -f1 | sed 's/>//')
            echo ">$seq_id" > "$class_dir/temp.fasta"
        else
            echo "$line" >> "$class_dir/temp.fasta"
            
            ./blastp -query "$class_dir/temp.fasta" -db "$db_path" \
                     -out "$class_dir/${seq_id}_blast_results.txt" \
                     -outfmt "6 qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore" \
                     -evalue 1e-5 -max_target_seqs 25 \
                     -num_threads 12

            if [ -s "$class_dir/${seq_id}_blast_results.txt" ]; then
                echo "Processing BLAST results for $seq_id"
                awk '$3 < 100' "$class_dir/${seq_id}_blast_results.txt" | \
                sort -k12,12nr | \
                head -n 10 > "$class_dir/${seq_id}_filtered_results.txt"
                
                if [ -s "$class_dir/${seq_id}_filtered_results.txt" ]; then
                    echo "Filtered results for $seq_id saved."
                else
                    echo "No filtered results for $seq_id."
                fi
            else
                echo "No BLAST results for $seq_id."
            fi
            
            current_sequence=$((current_sequence + 1))
            progress=$((current_sequence * 100 / total_sequences))
            show_progress $progress
        fi
    done < "$fasta_file"

    # Check if any filtered results exist before concatenating
    if [ -n "$(ls -A $class_dir/*_filtered_results.txt 2>/dev/null)" ]; then
        cat "$class_dir"/*_filtered_results.txt > "$class_dir/${filename}_all_filtered_results.txt"
        echo "Combined filtered results for $filename into ${filename}_all_filtered_results.txt"
    else
        echo "No filtered results found for $filename."
    fi

    # Cleanup temporary files
    rm "$class_dir/temp.fasta" 2>/dev/null
done

echo -e "\nBLAST searches completed."
