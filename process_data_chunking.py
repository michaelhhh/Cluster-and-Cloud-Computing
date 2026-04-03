import os
import json
import time
from mpi4py import MPI

# 1. Initialize MPI Environment
comm = MPI.COMM_WORLD
rank = comm.Get_rank()  # Unique ID for each process
size = comm.Get_size()  # Total number of processes

def process_byte_chunk(filename, start, end):
    """
    Processes a specific byte range of an NDJSON file to count languages.
    Each rank handles its own slice of the file to reduce I/O bottlenecks.
    """
    local_counts = {}
    
    try:
        with open(filename, 'rb') as f:
            # Syncing: If not the first chunk, skip the first partial line
            if start != 0:
                f.seek(start - 1)
                f.readline()
            
            # Process until the end of the assigned byte range
            while f.tell() < end:
                line = f.readline()
                if not line:
                    break
                
                try:
                    # Decode binary to string and parse JSON
                    data = json.loads(line.decode('utf-8'))
                    
                    # Identify container based on dataset type (Mastodon vs BlueSky)
                    container = data.get('doc') or data.get('record') or data
                    langs = container.get('language') or container.get('langs')
                    
                    # Count languages, handling lists for multi-lingual posts
                    if langs:
                        if isinstance(langs, list):
                            for l in langs:
                                if l: # Ensure language code is not Null
                                    local_counts[l] = local_counts.get(l, 0) + 1
                        else:
                            local_counts[langs] = local_counts.get(langs, 0) + 1
                            
                except (json.JSONDecodeError, UnicodeDecodeError, AttributeError):
                    # Skips ill-formatted lines or missing attributes
                    continue
                    
    except FileNotFoundError:
        if rank == 0:
            print(f"Error: {filename} not found.")
            
    return local_counts

if __name__ == "__main__":
    # Define files to process (ensure symbolic links are created on SPARTAN)
    target_files = ["mastodon-large.ndjson", "bluesky-large.ndjson"]
    total_final_counts = {}

    # Start Timing: Synchronize all ranks before starting the clock
    comm.Barrier()
    start_time = MPI.Wtime()

    for filename in target_files:
        # Rank 0 calculates the distribution offsets
        if rank == 0:
            if os.path.exists(filename):
                file_size = os.path.getsize(filename)
                chunk_size = file_size // size
                offsets = [(i * chunk_size, (i + 1) * chunk_size) for i in range(size)]
                offsets[-1] = (offsets[-1][0], file_size) # Ensure the last chunk reaches EOF
            else:
                offsets = [(0, 0)] * size
        else:
            offsets = None

        # 2. Distribute tasks using Scatter
        my_range = comm.scatter(offsets, root=0)
        
        # Each rank processes its specific chunk
        if my_range[1] > my_range[0]:
            local_data = process_byte_chunk(filename, my_range[0], my_range[1])
        else:
            local_data = {}
        
        # 3. Aggregate results back to Rank 0 using Gather
        all_results = comm.gather(local_data, root=0)

        # Rank 0 merges results for the current file
        if rank == 0:
            for partial_dict in all_results:
                for lang, count in partial_dict.items():
                    total_final_counts[lang] = total_final_counts.get(lang, 0) + count

    # 4. Final Output (Only on Rank 0)
    if rank == 0:
        end_time = MPI.Wtime()
        total_duration = end_time - start_time
        
        print("\n" + "="*45)
        print(f"FINAL EXECUTION REPORT")
        print("="*45)
        print(f"Total Execution Time: {total_duration:.4f} seconds")
        print("-" * 45)
        print(f"{'Language':<15} | {'Frequency (#posts)':<20}")
        print("-" * 45)
        
        # Sort by frequency descending and show Top 10
        sorted_langs = sorted(total_final_counts.items(), key=lambda x: x[1], reverse=True)
        for lang, count in sorted_langs[:10]:
            print(f"{lang:<15} | {count:<20,}")
            
        print("="*45)