import os
import subprocess
import argparse

def compress(input_dir, output_file, threads=8, compression_level=5):
    """
    Compress a folder using tar + zstd.
    """
    input_dir = os.path.abspath(input_dir)
    parent_dir = os.path.dirname(input_dir)
    base_name = os.path.basename(input_dir)

    print(f"Compressing: {input_dir} → {output_file}")
    cmd = [
        "tar",
        "-I", f"zstd -T{threads} -{compression_level}",
        "-cf", output_file,
        "-C", parent_dir,
        base_name
    ]
    subprocess.run(cmd, check=True)
    print("Compression complete.")

def decompress(archive_path, output_dir, threads=8):
    """
    Decompress a .tar.zst archive using tar + zstd.
    """
    archive_path = os.path.abspath(archive_path)
    output_dir = os.path.abspath(output_dir)

    print(f"Decompressing: {archive_path} → {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        "tar",
        "--use-compress-program", f"zstd -T{threads} -d",
        "-xf", archive_path,
        "-C", output_dir
    ]
    subprocess.run(cmd, check=True)
    print("Decompression complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fast tar+zstd compressor/decompressor")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Compress command
    compress_parser = subparsers.add_parser("compress")
    compress_parser.add_argument("--input_dir", required=True)
    compress_parser.add_argument("--output_file", default="output.tar.zst")
    compress_parser.add_argument("--threads", type=int, default=8)
    compress_parser.add_argument("--compression_level", type=int, default=5)

    # Decompress command
    decompress_parser = subparsers.add_parser("decompress")
    decompress_parser.add_argument("--archive_path", required=True)
    decompress_parser.add_argument("--output_dir", default=".")
    decompress_parser.add_argument("--threads", type=int, default=8)

    args = parser.parse_args()

    if args.command == "compress":
        compress(args.input_dir, args.output_file, args.threads, args.compression_level)
    elif args.command == "decompress":
        decompress(args.archive_path, args.output_dir, args.threads)
