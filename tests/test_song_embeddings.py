from ai_engineering_lab.song_embeddings.train_song_embeddings import (
    parse_playlists,
    parse_song_metadata,
    recommend_songs,
    train_song_model,
)


def test_parse_playlists_skips_metadata_and_single_song_rows():
    contents = "playlist count: 2\nsong count: 3\n0 1 2\n3\n4 5\n"
    assert parse_playlists(contents) == [["0", "1", "2"], ["4", "5"]]


def test_parse_song_metadata_maps_ids_to_title_and_artist():
    contents = "0\tFirst Song\tArtist One\n1\tSecond Song\tArtist Two\n"
    assert parse_song_metadata(contents) == {
        "0": {"title": "First Song", "artist": "Artist One"},
        "1": {"title": "Second Song", "artist": "Artist Two"},
    }


def test_recommend_songs_uses_song_ids_and_returns_metadata():
    playlists = [
        ["a", "b", "c"],
        ["a", "b", "d"],
        ["a", "b", "c"],
        ["x", "y", "z"],
    ]
    songs = {
        "a": {"title": "Song A", "artist": "Artist A"},
        "b": {"title": "Song B", "artist": "Artist B"},
        "c": {"title": "Song C", "artist": "Artist C"},
        "d": {"title": "Song D", "artist": "Artist D"},
        "x": {"title": "Song X", "artist": "Artist X"},
        "y": {"title": "Song Y", "artist": "Artist Y"},
        "z": {"title": "Song Z", "artist": "Artist Z"},
    }
    model = train_song_model(playlists, vector_size=8, window=2, epochs=5)

    recommendations = recommend_songs(model, songs, "a", topn=2)

    assert len(recommendations) == 2
    assert all(song_id != "a" for song_id, _, _ in recommendations)
    assert all(metadata["title"].startswith("Song ") for _, metadata, _ in recommendations)


def test_recommend_songs_rejects_unknown_id():
    model = train_song_model([["a", "b"], ["b", "c"]], vector_size=8, epochs=2)
    try:
        recommend_songs(model, {}, "missing")
    except ValueError as error:
        assert "not in the trained vocabulary" in str(error)
    else:
        raise AssertionError("Expected an unknown song ID to raise ValueError.")
