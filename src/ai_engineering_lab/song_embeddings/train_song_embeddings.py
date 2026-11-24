"""Train song embeddings from playlist co-occurrence using Word2Vec."""

from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import urlopen

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gensim.models import Word2Vec

PLAYLIST_URL = (
    "https://storage.googleapis.com/maps-premium/dataset/yes_complete/train.txt"
)
SONG_METADATA_URL = (
    "https://storage.googleapis.com/maps-premium/dataset/yes_complete/song_hash.txt"
)


def download_if_missing(url: str, destination: Path) -> Path:
    """Download a dataset file once and reuse the local cached copy."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        print(f"Downloading {destination.name}...")
        with urlopen(url, timeout=60) as response:
            destination.write_bytes(response.read())
    return destination


def parse_playlists(contents: str) -> list[list[str]]:
    """Parse playlist rows, skipping the two metadata lines and singletons."""
    lines = contents.splitlines()[2:]
    return [line.split() for line in lines if len(line.split()) > 1]


def parse_song_metadata(contents: str) -> dict[str, dict[str, str]]:
    """Map each song ID to its title and artist."""
    songs: dict[str, dict[str, str]] = {}
    for line in contents.splitlines():
        fields = line.split("\t")
        if len(fields) >= 3:
            song_id, title, artist = (field.strip() for field in fields[:3])
            songs[song_id] = {"title": title, "artist": artist}
    return songs


def load_dataset(cache_dir: Path | None = None):
    """Load the public playlist dataset and song metadata into memory."""
    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "ai-engineering-lab" / "song-embeddings"

    playlist_path = download_if_missing(PLAYLIST_URL, cache_dir / "train.txt")
    metadata_path = download_if_missing(SONG_METADATA_URL, cache_dir / "song_hash.txt")

    playlists = parse_playlists(playlist_path.read_text(encoding="utf-8"))
    songs = parse_song_metadata(metadata_path.read_text(encoding="utf-8"))
    return playlists, songs


def train_song_model(
    playlists: list[list[str]],
    *,
    vector_size: int = 32,
    window: int = 20,
    negative: int = 50,
    epochs: int = 10,
) -> Word2Vec:
    """Learn song vectors from songs appearing near one another in playlists."""
    from gensim.models import Word2Vec

    return Word2Vec(
        sentences=playlists,
        vector_size=vector_size,
        window=window,
        negative=negative,
        min_count=1,
        workers=1,
        seed=42,
        epochs=epochs,
    )


def recommend_songs(
    model: Word2Vec,
    songs: dict[str, dict[str, str]],
    song_id: str,
    *,
    topn: int = 5,
) -> list[tuple[str, dict[str, str], float]]:
    """Return similar song IDs, metadata and cosine-similarity scores."""
    if song_id not in model.wv:
        raise ValueError(f"Song ID {song_id!r} is not in the trained vocabulary.")

    results = []
    for similar_id, score in model.wv.most_similar(song_id, topn=topn):
        metadata = songs.get(
            similar_id, {"title": "Unknown title", "artist": "Unknown artist"}
        )
        results.append((similar_id, metadata, float(score)))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recommend songs using playlist-trained Word2Vec embeddings."
    )
    parser.add_argument("--song-id", default="2172", help="Song ID to find matches for.")
    parser.add_argument("--topn", type=int, default=5, help="Number of recommendations.")
    parser.add_argument("--vector-size", type=int, default=32)
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    if args.topn < 1 or args.epochs < 1:
        parser.error("--topn and --epochs must be positive integers.")

    playlists, songs = load_dataset()
    print(f"Loaded {len(playlists):,} playlists and {len(songs):,} songs.")

    selected_song = songs.get(args.song_id)
    if selected_song is None:
        parser.error(f"No metadata found for song ID {args.song_id!r}.")

    model = train_song_model(
        playlists,
        vector_size=args.vector_size,
        window=args.window,
        epochs=args.epochs,
    )

    print(f"\nRecommendations for: {selected_song['title']} — {selected_song['artist']}")
    for song_id, metadata, score in recommend_songs(
        model, songs, args.song_id, topn=args.topn
    ):
        print(
            f"{metadata['title']} — {metadata['artist']} "
            f"(ID: {song_id}, similarity: {score:.3f})"
        )


if __name__ == "__main__":
    main()
