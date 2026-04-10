import os
import json
import sys
from collections import Counter
from mpi4py import MPI


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def extract_languages(post):
    langs_found = []

    if isinstance(post.get("langs"), list):
        for lang in post["langs"]:
            if isinstance(lang, str) and lang.strip():
                langs_found.append(lang.strip().lower())

    elif isinstance(post.get("langs"), str) and post["langs"].strip():
        langs_found.append(post["langs"].strip().lower())

    elif isinstance(post.get("record"), dict):
        record = post["record"]

        if isinstance(record.get("langs"), list):
            for lang in record["langs"]:
                if isinstance(lang, str) and lang.strip():
                    langs_found.append(lang.strip().lower())

        elif isinstance(record.get("langs"), str) and record["langs"].strip():
            langs_found.append(record["langs"].strip().lower())

        elif isinstance(record.get("language"), str) and record["language"].strip():
            langs_found.append(record["language"].strip().lower())


    elif isinstance(post.get("doc"), dict):
        doc = post["doc"]

        if isinstance(doc.get("language"), str) and doc["language"].strip():
            langs_found.append(doc["language"].strip().lower())

        elif isinstance(doc.get("langs"), list):
            for lang in doc["langs"]:
                if isinstance(lang, str) and lang.strip():
                    langs_found.append(lang.strip().lower())

        elif isinstance(doc.get("langs"), str) and doc["langs"].strip():
            langs_found.append(doc["langs"].strip().lower())


    elif isinstance(post.get("language"), str) and post["language"].strip():
        langs_found.append(post["language"].strip().lower())

    return langs_found


def process_byte_chunk(filename, start, end):
    local_counts = Counter()
    total_lines = 0
    bad_json_lines = 0
    posts_with_language = 0
    posts_without_language = 0

    with open(filename, "rb") as f:
        if start != 0:
            f.seek(start - 1)
            f.readline()
        else:
            f.seek(start)

        while f.tell() < end:
            line = f.readline()
            if not line:
                break

            total_lines += 1

            try:
                post = json.loads(line.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                bad_json_lines += 1
                continue

            langs = extract_languages(post)

            if langs:
                posts_with_language += 1
                for lang in langs:
                    local_counts[lang] += 1
            else:
                posts_without_language += 1

    return {
        "counts": local_counts,
        "total_lines": total_lines,
        "bad_json_lines": bad_json_lines,
        "posts_with_language": posts_with_language,
        "posts_without_language": posts_without_language,
    }


def merge_results(results):
    final_counts = Counter()
    total_lines = 0
    bad_json_lines = 0
    posts_with_language = 0
    posts_without_language = 0

    for r in results:
        final_counts.update(r["counts"])
        total_lines += r["total_lines"]
        bad_json_lines += r["bad_json_lines"]
        posts_with_language += r["posts_with_language"]
        posts_without_language += r["posts_without_language"]

    return {
        "counts": final_counts,
        "total_lines": total_lines,
        "bad_json_lines": bad_json_lines,
        "posts_with_language": posts_with_language,
        "posts_without_language": posts_without_language,
    }


def print_report(filename, result, elapsed_time, top_n=20):
    counts = result["counts"]

    print("=" * 60)
    print(f"FILE: {filename}")
    print("=" * 60)
    print(f"Total lines read:           {result['total_lines']}")
    print(f"Bad JSON lines skipped:     {result['bad_json_lines']}")
    print(f"Posts with language:        {result['posts_with_language']}")
    print(f"Posts without language:     {result['posts_without_language']}")
    print(f"Unique language codes:      {len(counts)}")
    print()
    print(f"Top {top_n} language counts:")

    for lang, freq in counts.most_common(top_n):
        print(f"{lang:<15} {freq}")

    print()
    print(f"Elapsed time for {filename}: {elapsed_time:.6f} seconds")
    print()


def get_ranges(filename, nprocs):
    file_size = os.path.getsize(filename)
    chunk_size = file_size // nprocs
    ranges = []

    for i in range(nprocs):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < nprocs - 1 else file_size
        ranges.append((start, end))

    return ranges


def process_file_parallel(filename):
    if rank == 0:
        ranges = get_ranges(filename, size)
    else:
        ranges = None

    my_range = comm.scatter(ranges, root=0)

    start, end = my_range
    local_result = process_byte_chunk(filename, start, end)

    all_results = comm.gather(local_result, root=0)

    if rank == 0:
        return merge_results(all_results)
    return None


def main():
    if len(sys.argv) < 2:
        if rank == 0:
            print("Usage:")
            print("  mpirun -np <n> python mpi_language_counter.py <file1> [file2] ...")
            print("Examples:")
            print("  mpirun -np 2 python mpi_language_counter.py mastodon-small.ndjson bluesky-small.ndjson")
            print("  mpirun -np 8 python mpi_language_counter.py mastodon-large.ndjson bluesky-large.ndjson")
        sys.exit(1)

    filenames = sys.argv[1:]

    comm.Barrier()
    overall_start = MPI.Wtime()

    for filename in filenames:
        comm.Barrier()
        file_start = MPI.Wtime()

        result = process_file_parallel(filename)

        comm.Barrier()
        file_end = MPI.Wtime()

        if rank == 0:
            print_report(filename, result, file_end - file_start, top_n=20)

    comm.Barrier()
    overall_end = MPI.Wtime()

    if rank == 0:
        print("#" * 60)
        print(f"Total program elapsed time: {overall_end - overall_start:.6f} seconds")
        print("#" * 60)


if __name__ == "__main__":
    main()
