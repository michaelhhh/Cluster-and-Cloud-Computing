import os
import json
from mpi4py import MPI

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Tags for point-to-point communication
WORK_TAG = 1
DIE_TAG = 2

def worker_process(filename):
    """Worker loop: Requests work, processes chunks, and sends back data."""
    local_counts = {}
    while True:
        # 1. Ask Master for a task
        comm.send(None, dest=0, tag=WORK_TAG)
        
        # 2. Receive task (start_byte, end_byte) or DIE signal
        status = MPI.Status()
        task = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
        
        if status.Get_tag() == DIE_TAG:
            break
            
        # 3. Process the assigned byte chunk
        start, end = task
        with open(filename, 'rb') as f:
            if start != 0:
                f.seek(start - 1)
                f.readline() # Sync to next full line
            
            while f.tell() < end:
                line = f.readline()
                if not line: break
                try:
                    data = json.loads(line.decode('utf-8'))
                    container = data.get('doc') or data.get('record') or data
                    langs = container.get('language') or container.get('langs')
                    if langs:
                        if isinstance(langs, list):
                            for l in langs:
                                local_counts[l] = local_counts.get(l, 0) + 1
                        else:
                            local_counts[langs] = local_counts.get(langs, 0) + 1
                except: continue
                
    # Send final local counts back to Master for merging
    comm.send(local_counts, dest=0, tag=WORK_TAG)

def master_process(filenames):
    """Master loop: Manages file offsets and aggregates final results."""
    total_counts = {}
    chunk_size = 10 * 1024 * 1024  # 10MB tasks
    
    for filename in filenames:
        file_size = os.path.getsize(filename)
        current_pos = 0
        active_workers = size - 1
        
        while active_workers > 0:
            # Wait for any worker to request work
            status = MPI.Status()
            comm.recv(source=MPI.ANY_SOURCE, tag=WORK_TAG, status=status)
            worker_id = status.Get_source()
            
            if current_pos < file_size:
                # Send the next byte range task
                next_pos = min(current_pos + chunk_size, file_size)
                comm.send((current_pos, next_pos), dest=worker_id, tag=WORK_TAG)
                current_pos = next_pos
            else:
                # No more work for this file; send DIE signal
                comm.send(None, dest=worker_id, tag=DIE_TAG)
                # Receive final dictionary from the finishing worker
                worker_data = comm.recv(source=worker_id, tag=WORK_TAG)
                for lang, count in worker_data.items():
                    total_counts[lang] = total_counts.get(lang, 0) + count
                active_workers -= 1
                
    return total_counts

if __name__ == "__main__":
    target_files = ["mastodon-large.ndjson", "bluesky-large.ndjson"]
    
    comm.Barrier()
    start_time = MPI.Wtime()
    
    if rank == 0:
        # Master manages the queue and prints final results
        final_results = master_process(target_files)
        duration = MPI.Wtime() - start_time
        
        print(f"Master-Worker Total Time: {duration:.4f} seconds")
        sorted_langs = sorted(final_results.items(), key=lambda x: x[1], reverse=True)
        for lang, count in sorted_langs[:10]:
            print(f"{lang}: {count:,}")
    else:
        # Workers process each file in the list
        for filename in target_files:
            worker_process(filename)