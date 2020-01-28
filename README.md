# face2emoji
generate emoji by using conditional DCGAN &amp; convert face photos into emojis by using CNN and encoder-decoder (captioner)

## description about files
- cgan_emoji.py : conditional DCGAN
- face_classifier.py : used to train CNN to classify faces to 8 emotions
- generate_emoji_from_photo.py : convert representation obtained from face photo into emojis
- get_inception_score.py : calculate inception score of a batch of images
- load_photo.py : load photographs, then get emotion vectors consist of 8 feelings and get embedding correspond to each face photo
- make_caption.py : used to get 300-dim Embedding (converted from descriptions of each emoji) directly from face photo
- make_dataset_of_photograph.py : load photographs (e.g. jpeg) and convert them to NumPy Array
- make_emoji_from_vector.py : convert vector into emoji
- preprocess_dataset.py : used to preprocess dataset (emojis and texts)

## note
- please install pretrained pretrained model into the `utils` directory ( -> `utils/glove.6B.300d.txt`)
  - download `glove.6B.zip` from [here](https://nlp.stanford.edu/projects/glove/)
- directory on which generated emojis are stored or image files (i.e. generated emojis or my photographs) are omitted from this git. so when you encount some errors because of File Not Found, please create such directories and designate the path.