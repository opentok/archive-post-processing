# archive-stitcher

**Merge two consecutive overlapping archives into one archive**


## Installing prerequisites

- Install git-lfs before checking out the code

In linux
```sh
apt-get install -y git-lfs
```

In mac
```sh
brew install git-lfs
```

Enable git-lfs
```sh
git lfs install
```

- Set virtualenv

```sh
python -m venv venv
source venv/bin/activate
```

- Install src dependencies
```sh
pip install -r requirements.txt
```

- Install test dependencies
```sh
pip install -r requirements-test.txt
```

## Running  the code

- Running the code
```sh
python -m src.archive_stitcher
```

- Get the current list of parameters
```sh
python -m src.archive_stitcher --help
```

- Example
```sh
python -m src.archive_stitcher -a <mp4_archive_a_path> -b <mp4_archive_b_path> -o <mp4_output_archive_path> -k mse -g pearson -x <assessment_time_in_seconds> -y -s
```

## Running tests and linter

- Running tests
```sh
python -m pytest test
```

- Running tests with filter
```sh
python -m pytest test -k <partial match to test name>
```

- Running tests with verbose output
```sh
python -m pytest test -sv
```

- Running linter
```sh
python -m pylint src test
```

- Run manual sanity check for merge code

> Run test without deleting the output
```sh
mkdir -p /tmp/output
TEST_OUTPUT=/tmp/output python -m pytest test -vs -k test_merge_success_given_overlap
```

> Look for `output.mp4` in a the most recent directory inside `/tmp/output`

> Compare seconds 16-25 with [output_ref.mp4](test_data/screenshare_low_variation/output_ref.mp4)

## Media format

The input media format we currently use is the following:

* Video
    - codec: H264
    - profile: Constrained Baseline
    - pixel format: YUV 420p
    - level: 4.1
    - FPS: 25
    - PTS timescale: 90000
* Audio
    - codec: AAC LC
    - sample rate: 48Khz
    - channels: 1
* Container
    - type: mp4

### AAC priming

When developing the code that merges two audio files, we found that the FFMPEG filter `compat` was outputting AAC audio
data that belongs to its *priming* and was not audible in the original files. This was causing audio/video desync in
the output file.

To avoid this we find out the number of audio samples used for priming from the original files and we then apply
an offset to the audio in the output file. Otherwise we would have to re-encode the audio.

For more information about AAC priming see this [link](https://developer.apple.com/documentation/quicktime-file-format/background_aac_encoding)



### Tool for generating test data

You can generate test data using the `tools.sample_generator` module. This tool generates a video file
that gets cut into two overlapping parts.

You can add effects such as glitches, audio video desync in the second part and black frames at each end of the overlap.


- Example command

```sh
python -m tools.sample_generator -a test_data/screenshare_low_variation/output_ref.mp4 -l 120 -L 120 -o 5 -O 10 -d output
```

- Help output
```sh
python -m tools.sample_generator --help

usage: sample_generator.py [-h] -a ARCHIVE -l MIN_LENGTH -L MAX_LENGTH -o MIN_OVERLAP -O MAX_OVERLAP -d OUTPUT_PREFIX [-t {full,audio_only,video_only,video_with_silent_audio,audio_with_static_video,no_overlap}] [-e EFFECTS]
                           [--random-seed RANDOM_SEED] [--av-sync-difference AV_SYNC_DIFFERENCE]

options:
  -h, --help            show this help message and exit
  -a, --archive ARCHIVE
  -l, --min-length MIN_LENGTH
                        minimum length of the generated sample in seconds, allows decimals
  -L, --max-length MAX_LENGTH
                        maximum length of the generated sample in seconds, allows decimals
  -o, --min-overlap MIN_OVERLAP
                        minimum overlap length in seconds, allows decimals
  -O, --max-overlap MAX_OVERLAP
                        maximum overlap length in seconds, allows decimals
  -d, --output-prefix OUTPUT_PREFIX
  -t, --sample-type {full,audio_only,video_only,video_with_silent_audio,audio_with_static_video,no_overlap}
                        Type of sample to generate (default: full)
  -e, --effect EFFECTS  Effects to apply (can be specified multiple times, allowed: ['black_frame_at_end_of_a', 'black_frame_at_ini_of_b', 'glitch_at_end_of_a', 'glitch_at_ini_of_b'])
  --random-seed RANDOM_SEED
                        Random seed for reproducibility (default: None)
  --av-sync-difference AV_SYNC_DIFFERENCE
                        Difference in audio/video sync between the two outputs in seconds, allows decimals
```
