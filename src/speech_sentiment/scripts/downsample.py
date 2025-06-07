import os
import argparse
import random
import csv
import sox
from tqdm import tqdm
from multiprocessing.pool import ThreadPool

def process_file(row, clips_directory, directory, output_format):
    file_name = row['path']
    clips_name = os.path.basename(os.path.splitext(file_name)[0]) + '.' + output_format
    sentiment = row['emotion']
    audio_path = os.path.join(directory, file_name)
    output_audio_path = os.path.join(clips_directory, clips_name)
    
    # Create necessary directories for the output path
    os.makedirs(os.path.dirname(output_audio_path), exist_ok=True)

    # Convert to FLAC or WAV using Sox and downsample to 16kHz
    tfm = sox.Transformer()
    tfm.rate(samplerate=16000)
    tfm.build(input_filepath=audio_path, output_filepath=output_audio_path)

    return {'path': output_audio_path, 'emotion': sentiment}

def downsample(file_path, output_dir, num_workers, output_format='flac'):
    data = []
    directory = os.path.dirname(file_path)

    clips_directory = os.path.abspath(os.path.join(output_dir, 'downsampled_clips'))

    if not os.path.exists(clips_directory):
        os.makedirs(clips_directory)

    with open(file_path, encoding="utf-8") as f:
        length = sum(1 for _ in f) - 1

    with open(file_path, newline='', encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file, delimiter=',')
        data_to_process = [(row, clips_directory, directory, output_format) for row in reader]

    print(f"{length} files found. Converting to {output_format.upper()} using {num_workers} workers.")
    with ThreadPool(num_workers) as pool:
        data = list(tqdm(pool.imap(lambda x: process_file(*x), data_to_process), total=length))

    # Saving the processed data into a single CSV file
    csv_path = os.path.join(output_dir, 'emotion_dataset.csv')

    with open(csv_path, mode='w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['path', 'emotion'])
        writer.writeheader()
        writer.writerows(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=
                                     """
                                        Utility script to downsample speech sentiment datasets.
                                     """
                                    )
    parser.add_argument('--file_path', type=str, default=None, required=True,
                        help='path to one of the .tsv files found in cv-corpus')
    parser.add_argument('--output_dir', type=str, default=None, required=True,
                        help='path to the dir where the downsampled clips are supposed to be saved')
    parser.add_argument('-w', '--num_workers', type=int, default=2,
                        help='number of worker threads for processing')
    parser.add_argument('--output_format', type=str, default='flac',
                        help='output audio format (flac or wav)')

    args = parser.parse_args()
    downsample(**args)
