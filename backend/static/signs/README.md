# Sign Language Video Assets

Place MP4 video files here to enable animated sign playback.

## Directory structure

```
static/signs/
  HELLO.mp4          ← whole-word sign videos (uppercase filenames)
  THANK.mp4
  SORRY.mp4
  ...
  letters/
    A.mp4            ← fingerspelling letter videos (A–Z)
    B.mp4
    ...
    Z.mp4
```

## How the system works

1. User types text → backend tokenises it (NLTK lemmatisation + tense detection)
2. For each token the backend checks if `static/signs/<TOKEN>.mp4` exists
3. If found → video URL is returned → Flutter plays it
4. If not found → letters are spelled out using `static/signs/letters/<X>.mp4`
5. If no letter video either → animated text card is shown as fallback

## Video source

Download ISL (Indian Sign Language) pre-rendered MP4s from:
  https://github.com/jigargajjar55/Audio-Speech-To-Sign-Language-Converter

Copy the video files from that repo's `isl_gifs/` or `static/` folder here,
renaming them to UPPERCASE with .mp4 extension.

Example rename:
  hello.mp4  →  HELLO.mp4
  thank_you.mp4  →  THANK.mp4   (or THANK_YOU.mp4 — matches the token exactly)

## Without videos

The app works immediately without any video files.
Each sign is shown as a large animated text card with the word/letter displayed.
Add videos progressively to enhance the experience.
