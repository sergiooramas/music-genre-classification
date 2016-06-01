# Music Genre Classification Dataset

A subset of the <a href="http://mtg.upf.edu/download/datasets/mard">MARD dataset</a> was created for genre classification experiments. It contains 100 albums by genre from different artists, from 13 different genres (Alternative Rock, Classical, Country, Dance & Electronic, Folk, Jazz, Latin Music, Metal, New Age, Pop, R&B, Rap & Hip-Hop, Rock). All the albums have been mapped to MusicBrainz and AcousticBrainz. It contains linguistic and sentiment features. It is stored as a dictionary, where the keys are the amazon-ids. The file is called <b>classification_dataset.json</b>

We also provide all the necessary files to reproduce the experiments on genre classification in the paper referenced below. entity_features_dataset.json contains the entities and categories identified in the reviews for every album, entity_features_dataset_broader.json contains also the broader Wikipedia categories, genre_classification.py is the Python script used for the experiment. Finally, train_x.csv and test_x.csv contains the 5 different splits in the dataset used for cross validation.

## References

If you use this code for research purposes, please cite our <a target="_blank" href="http://mtg.upf.edu/node/3490">paper</a>:

Oramas, S., Espinosa-Anke L., Lawlor A., Serra X., & Saggion H. (2016). Exploring Customer Reviews for Music Genre Classification and Evolutionary Studies. 17th International Society for Music Information Retrieval Conference (ISMIR16).

## License

This project is licensed under the terms of the MIT license.
