import json
import time
from mpi4py import MPI

# 1. Initialize MPI Environment
comm = MPI.COMM_WORLD
rank = comm.Get_rank()  # Unique ID for each core
size = comm.Get_size()  # Total number of cores requested in SLURM

def count_languages(filenames):
    """Parses NDJSON files and aggregates language counts."""
    local_counts = {}
    
    for filename in filenames:
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                # 2. Parallelization: Each rank processes a subset of lines
                for i, line in enumerate(f):
                    if i % size == rank:
                        try:
                            data = json.loads(line)
                            
                            # 3. Identify Container: 'doc' (Mastodon) or 'record' (BlueSky)
                            # This handles the specific nesting discovered via head -n 1
                            container = data.get('doc') or data.get('record') or data
                            
                            # 4. Extract Attribute: 'language' or 'langs'
                            langs = container.get('language') or container.get('langs')
                            
                            # 5. Handle nulls, strings, and lists
                            if langs:
                                if isinstance(langs, list):
                                    for l in langs:
                                        local_counts[l] = local_counts.get(l, 0) + 1
                                else:
                                    local_counts[langs] = local_counts.get(langs, 0) + 1
                                    
                        except (json.JSONDecodeError, AttributeError):
                            # Skips ill-formatted lines
                            continue 
        except FileNotFoundError:
            if rank == 0:
                print(f"Error: {filename} not found. Check your symbolic links.")
                
    return local_counts

if __name__ == "__main__":
    # 6. Start Timing
    # We synchronize to ensure all ranks start timing at the same moment
    comm.Barrier()
    start_time = MPI.Wtime()

    # Define the files to process (Update these to the 'large' files for your final runs)
    #
    target_files = ["mastodon-medium.ndjson", "bluesky-medium.ndjson"]
    
    # Each core counts its assigned lines locally
    my_local_data = count_languages(target_files)

    # 7. Aggregate Results: Gather all dictionaries to Rank 0
    all_results = comm.gather(my_local_data, root=0)

    if rank == 0:
        # Merge dictionaries from all cores into one final result
        final_counts = {}
        for partial_dict in all_results:
            for lang, count in partial_dict.items():
                final_counts[lang] = final_counts.get(lang, 0) + count
        
        # 8. End Timing and Output Results
        end_time = MPI.Wtime()
        total_duration = end_time - start_time
        
        print("-" * 30)
        print(f"Total Execution Time: {total_duration:.4f} seconds")
        print("-" * 30)
        print("Language Frequency Table (Top 10):")
        # Sort by frequency descending
        sorted_langs = sorted(final_counts.items(), key=lambda x: x[1], reverse=True)
        for lang, count in sorted_langs[:10]:
            print(f"{lang}: {count}")
        print("-" * 30)