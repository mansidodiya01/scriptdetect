# Copyright 2023 The MediaPipe Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0

import argparse
import time
from mediapipe.tasks import python
from mediapipe.tasks.python.audio.core import audio_record
from mediapipe.tasks.python.components import containers
from mediapipe.tasks.python import audio

def run(model: str, max_results: int, score_threshold: float) -> None:
    """Continuously run inference on audio data acquired from the device.

    Args:
        model: Name of the TFLite audio classification model.
        max_results: Maximum number of classification results to display.
        score_threshold: The score threshold of classification results.
    """

    interval_between_inference = 0.35  # Match frame capture timing in seconds

    # Callback to process classification results.
    def save_result(result: audio.AudioClassifierResult, timestamp_ms: int):
        detected_time = time.time()  # High-precision timestamp
        detected = False
        for category in result.classifications[0].categories:
            if ("baby cry" in category.category_name.lower() or
                "infant cry" in category.category_name.lower()) and category.score > score_threshold:
                print(f"[{detected_time:.3f}] Baby is crying! Detected: {category.category_name}, Confidence: {category.score:.2f}")
                detected = True

    # Initialize the audio classification model.
    base_options = python.BaseOptions(model_asset_path=model)
    options = audio.AudioClassifierOptions(
        base_options=base_options,
        running_mode=audio.RunningMode.AUDIO_STREAM,
        max_results=max_results,
        score_threshold=score_threshold,
        result_callback=save_result,
    )
    classifier = audio.AudioClassifier.create_from_options(options)

    # Initialize the audio recorder and a tensor to store the audio input.
    buffer_size, sample_rate, num_channels = 15600, 16000, 1
    audio_format = containers.AudioDataFormat(num_channels, sample_rate)
    record = audio_record.AudioRecord(num_channels, sample_rate, buffer_size)
    audio_data = containers.AudioData(buffer_size, audio_format)

    # Start audio recording in the background.
    record.start_recording()

    # Loop until interrupted.
    print("Starting audio classification. Press Ctrl+C to stop.")
    while True:
        try:
            # Load the input audio from the AudioRecord instance and run classify.
            data = record.read(buffer_size)
            audio_data.load_from_array(data)
            classifier.classify_async(audio_data, time.time_ns() // 1_000_000)

            # Throttle the inference to match frame capture timing
            time.sleep(interval_between_inference)
        except KeyboardInterrupt:
            print("Stopping audio classification...")
            break

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--model',
        help='Name of the audio classification model.',
        required=False,
        default='yamnet.tflite',
    )
    parser.add_argument(
        '--maxResults',
        help='Maximum number of results to show.',
        required=False,
        default=5,
    )
    parser.add_argument(
        '--scoreThreshold',
        help='The score threshold of classification results.',
        required=False,
        default=0.8,
    )
    args = parser.parse_args()

    run(args.model, int(args.maxResults), float(args.scoreThreshold))

if __name__ == '__main__':
    main()
