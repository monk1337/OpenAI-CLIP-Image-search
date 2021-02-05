import clip
import torch
import pandas as pd
import json
import numpy as np
from urllib.request import urlopen

# Load the open CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Load the photo IDs
photo_ids = pd.read_csv("/Users/monk/Desktop/research_paper_visuals/clip_model_api/unsplash/photo_ids.csv")
photo_ids = list(photo_ids['photo_id'])

# Load the features vectors
photo_features = np.load("/Users/monk/Desktop/research_paper_visuals/clip_model_api/unsplash/features.npy")

# Convert features to Tensors: Float32 on CPU and Float16 on GPU
if device == "cpu":
    photo_features = torch.from_numpy(photo_features).float().to(device)
else:
    photo_features = torch.from_numpy(photo_features).to(device)
# Print some statistics


def encode_search_query(search_query):
    with torch.no_grad():
        # Encode and normalize the search query using CLIP
        text_encoded = model.encode_text(clip.tokenize(search_query).to(device))
        text_encoded /= text_encoded.norm(dim=-1, keepdim=True)

    # Retrieve the feature vector
    return text_encoded


def find_best_matches(text_features, results_count=3, key = None):
    
    text_features        = encode_search_query(text_features)
    similarities         = (photo_features @ text_features.T).squeeze(1)
    best_photo_idx       = (-similarities).argsort()
    pics_data            = [photo_ids[i] for i in best_photo_idx[:results_count]]
    unsplash_api_url     = [f"https://api.unsplash.com/photos/{pic}?client_id={str(key)}" for pic in pics_data]
    print(unsplash_api_url)
    photo_data           = [json.loads(urlopen(k).read().decode("utf-8")) for k in unsplash_api_url]
    return photo_data
