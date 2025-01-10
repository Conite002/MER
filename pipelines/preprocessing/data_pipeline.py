import os
import json
from tqdm import tqdm
import pandas as pd

from sklearn.decomposition import PCA
import pandas as pd
from preprocessing.audio.preprocess_audio import preprocess_audio_for_model
from preprocessing.text.preprocess_text import preprocess_text_for_model
from preprocessing.video.preprocess_video import preprocess_video_for_model

def reduce_dimensions(embeddings, target_dim):
    if embeddings is None:
        return None
    pca = PCA(n_components=target_dim)
    return pca.fit_transform(embeddings)

def process_single_video(vid, video_speakers, video_labels, video_audio, video_visual, video_sentence, text_tokenizer, text_model, target_dims):
    """
    Process a single video and prepare its multimodal features.

    Args:
        vid (str): Video ID.
        video_speakers (dict): Speakers metadata for the video.
        video_labels (dict): Labels for the video.
        video_audio (dict): Raw audio features for the video.
        video_visual (dict): Raw visual features for the video.
        video_sentence (dict): Text sentences for the video.
        text_tokenizer: Text tokenizer for encoding text.
        text_model: Text model for generating embeddings.
        target_dims (dict): Target dimensions for each modality.

    Returns:
        dict: Structured multimodal features for the video.
    """
    try:
        audio_features = preprocess_audio_for_model(video_audio[vid], target_dim=target_dims["audio"])

        visual_features = preprocess_video_for_model(video_visual[vid], target_dim=target_dims["visual"])

        sentence = video_sentence[vid]
        text_features = preprocess_text_for_model(sentence, text_tokenizer, text_model, max_length=target_dims["text"])

        return {
            "vid": vid,
            "speakers": video_speakers[vid],
            "labels": video_labels[vid],
            "audio": audio_features,
            "visual": visual_features,
            "text": text_features,
            "sentence": sentence
        }
    except Exception as e:
        print(f"Error processing video {vid}: {e}")
        return None

def prepare_dataset(video_ids, video_speakers, video_labels, video_audio, video_visual, video_sentence, text_tokenizer, text_model, target_dims):
    """
    Prepare a dataset for multimodal training/testing.

    Args:
        video_ids (list): List of video IDs.
        video_speakers (dict): Speakers metadata for each video.
        video_labels (dict): Labels for each video.
        video_audio (dict): Raw audio features for each video.
        video_visual (dict): Raw visual features for each video.
        video_sentence (dict): Text sentences for each video.
        text_tokenizer: Text tokenizer for encoding text.
        text_model: Text model for generating embeddings.
        target_dims (dict): Target dimensions for each modality.

    Returns:
        list: Structured dataset with multimodal features for each video.
    """
    dataset = []
    for vid in tqdm(video_ids, desc="Processing videos"):
        data = process_single_video(vid, video_speakers, video_labels, video_audio, video_visual, video_sentence, text_tokenizer, text_model, target_dims)
        if data is not None:
            dataset.append(data)
    return dataset

def save_dataset(dataset, output_path):
    """
    Save the processed dataset to a JSON file.

    Args:
        dataset (list): Processed dataset.
        output_path (str): Path to save the dataset.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=4)

def run_data_pipeline(train_vids, dev_vids, test_vids, video_speakers, video_labels, video_audio, video_visual, video_sentence, text_tokenizer, text_model):
    """
    Run the data preprocessing pipeline for train, dev, and test datasets.

    Args:
        train_vids, dev_vids, test_vids: Video IDs for train, dev, and test sets.
        video_speakers, video_labels, video_audio, video_visual, video_sentence: Multimodal data dictionaries.
        text_tokenizer, text_model: Tokenizer and model for text embeddings.
    """
    target_dims = {
        "audio": 100,
        "text": 768,
        "visual": 512
    }

    train_data = prepare_dataset(train_vids, video_speakers, video_labels, video_audio, video_visual, video_sentence, text_tokenizer, text_model, target_dims)
    dev_data = prepare_dataset(dev_vids, video_speakers, video_labels, video_audio, video_visual, video_sentence, text_tokenizer, text_model, target_dims)
    test_data = prepare_dataset(test_vids, video_speakers, video_labels, video_audio, video_visual, video_sentence, text_tokenizer, text_model, target_dims)

    save_dataset(train_data, "outputs/preprocessed/train_data.json")
    save_dataset(dev_data, "outputs/preprocessed/dev_data.json")
    save_dataset(test_data, "outputs/preprocessed/test_data.json")


def generate_metadata(csv_file, video_audio_metadata, output_json_path, dataset_type="train"):
    """
    Generate metadata for train/dev/test splits.

    Args:
        csv_file (str): Path to the CSV file containing text and labels.
        video_audio_metadata (list): List of video-audio metadata dictionaries.
        output_json_path (str): Path to save the generated JSON.
        dataset_type (str): Type of dataset ('train', 'test', 'dev').

    Returns:
        None
    """


    print(f"Generating metadata for {dataset_type}...")
    data = pd.read_csv(csv_file)
    metadata = []

    video_audio_map = {os.path.normpath(item["video_path"]): os.path.normpath(item["audio_path"]) for item in video_audio_metadata}
    print("First few video_audio_map keys:", list(video_audio_map.keys())[:5])

    if dataset_type == "train":
        base_video_dir = "../data/MELD.Raw/train/train_splits"
    elif dataset_type == "test":
        base_video_dir = "../data/MELD.Raw/test/output_repeated_splits_test"
    elif dataset_type == "dev":
        base_video_dir = "../data/MELD.Raw/dev/dev_splits_complete"
    else:
        raise ValueError(f"Invalid dataset_type: {dataset_type}")

    for _, row in tqdm(data.iterrows(), total=len(data), desc=f"Processing {dataset_type} metadata"):
        video_name = f"dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}.mp4"
        video_path = os.path.normpath(os.path.join(base_video_dir, video_name))

        if video_path in video_audio_map:
            metadata.append({
                "video": video_path,
                "audio": video_audio_map[video_path],
                "text": row["Utterance"],
                "label": row["Emotion"]
            })
        else:
            print(f"Warning: Video file {video_path} not found in audio metadata map.")

    if not metadata:
        print(f"No matching entries found for {dataset_type}. Check video_audio_metadata and CSV file.")

    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, "w") as f:
        json.dump(metadata, f, indent=4)
