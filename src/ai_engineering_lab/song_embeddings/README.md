# Song embeddings and recommendations

Train a Word2Vec model on playlists and use the resulting song embeddings to find songs that appear in similar playlist contexts.

## Reference

Jay Alammar and Maarten Grootendorst, [Hands-On Large Language Models](https://www.oreilly.com/library/view/hands-on-large-language/9781098150952/), section **Training a Song Embedding Model** (the book's example uses the public Million Playlist-style `yes_complete` dataset).

## How it works

The experiment treats each playlist as a sentence and each song ID as a word:

1. Download playlist sequences and a separate song-ID-to-title/artist metadata file.
2. Train Gensim's Word2Vec model on the playlist sequences.
3. Word2Vec learns a dense vector (embedding) for each song ID based on which other songs occur nearby in playlists.
4. Given a song ID, find the nearest song vectors using cosine similarity and display their metadata.

This is **collaborative filtering through co-occurrence**, not audio analysis. The model does not listen to the tracks or understand their lyrics, genre or musical structure. Similarity means that songs tend to occur in similar playlist contexts in this dataset; it does not guarantee that they sound alike.

The implementation keeps the book's core approach and key hyperparameters (32 dimensions, a context window of 20 and 50 negative samples), while making a few practical changes:

- Downloads the data on first run and caches it outside the repository in `~/.cache/ai-engineering-lab/song-embeddings/`.
- Uses a fixed random seed and one worker to make training more repeatable.
- Uses a small command-line interface instead of a hard-coded song ID.
- Looks up metadata by song ID. IDs are identifiers, not row positions, so dictionary lookup (or a DataFrame index lookup) is safer than using `iloc`.
- Prints similarity scores with each recommendation.

## How to run

Complete the [shared setup](../../../../README.md#shared-setup), then install this experiment's optional dependency from the repository root:

```bash
python -m pip install -e ".[song-embeddings]"
```

Run the default example (Metallica's **Fade to Black**, song ID `2172`):

```bash
python -m ai_engineering_lab.song_embeddings.train_song_embeddings
```

Choose another song ID or change the number of recommendations:

```bash
python -m ai_engineering_lab.song_embeddings.train_song_embeddings --song-id 2172 --topn 10
```

The first run downloads the dataset and trains the model, which may take a few minutes. Subsequent runs reuse the downloaded files but train the model again. No GPU or paid API key is required.

Useful options:

| Argument | Default | Purpose |
| --- | ---: | --- |
| `--song-id` | `2172` | Song ID to find similar songs for. |
| `--topn` | `5` | Number of recommendations. |
| `--vector-size` | `32` | Number of dimensions in each song embedding. |
| `--window` | `20` | Maximum context distance within each playlist. |
| `--epochs` | `10` | Number of training passes over the playlists. |

## Try next

- Compare recommendations for songs you know well and see whether the results match your expectations.
- Change `--window` and observe how the neighbourhood of a song changes.
- Consider how playlist order and playlist length influence the learned context.
- Later, compare these co-occurrence embeddings with text or audio embeddings. Those would capture different notions of similarity.

## Data and limitations

The dataset is downloaded from Google's public `yes_complete` dataset endpoints:

- [Playlist sequences](https://storage.googleapis.com/maps-premium/dataset/yes_complete/train.txt)
- [Song metadata](https://storage.googleapis.com/maps-premium/dataset/yes_complete/song_hash.txt)

The experiment relies on the contents and availability of those external files. It is intended for learning and experimentation, not as a production music recommender. Recommendations can reflect popularity, playlist construction habits and dataset biases.
