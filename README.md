# OpenAI-CLIP-Image-search
OpenAI's CLIP neural network

### Step 1
Install all the dependencies from dependencies.txt

Get a key from unsplash https://unsplash.com/oauth/applications

Download photo_ids and features

```shell
!wget https://transfer.army/api/download/TuWWFTe2spg/EDm6KBjc -O unsplash-dataset/photo_ids.csv
!wget https://transfer.army/api/download/LGXAaiNnMLA/AamL9PpU -O unsplash-dataset/features.npy
```

Change the path of photo_ids, photo_features in load_model.py


### Step 2

Run python3 clip_api.py
A server will start at http://0.0.0.0:5002/

### Step 3
Query using http://0.0.0.0:5002/get_url by passing three args 

```python
{"query" : {"text_query": "two pandas are running", "n": 3, "key": "azSDMk4G"}}
```
