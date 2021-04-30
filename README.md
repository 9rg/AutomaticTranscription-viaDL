# Automatic Transcription via DL

Transcript Japanese instruments(Japanese flute and Japanese drum) automatically with deep learning.

## DEMO

![ATDL_image](https://user-images.githubusercontent.com/53912472/116381796-31209100-a850-11eb-9c77-0ba15b0d0962.png)


## Features

The first attempt of using deep learning to unsteady Japanese festival music.
This time, the program handle "Awaodori festival music" as a target.

## Usage

```
# train model
python train.py -c config.yml

# generate original music data in csv format
python predict -c config.yml
```

## Note

~~This is the part of program I used in my graduation research in university.~~