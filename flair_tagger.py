import pandas as pd
from flair.data import Sentence
from flair.models import SequenceTagger

tagger = SequenceTagger.load("flair/ner-german-large")

df = pd.read_csv("22_10_22_hls_people_hist_hub_cleaned.csv", sep=";", nrows=10)


def predict_sentence(row):
    sentence = Sentence(row["text"])
    tagger.predict(sentence)
    return sentence.get_spans("ner")


# apply predict_sentence to each row
df["flair_tags"] = df.apply(predict_sentence, axis=1)

# extract the location tags from the flair tags and add them to the dataframe as a comma separated string
df["flair_locations"] = df["flair_tags"].apply(
    lambda x: ",".join([tag.text for tag in x if tag.tag == "LOC"])
)

# extract the person tags from the flair tags and add them to the dataframe as a comma separated string
df["flair_person"] = df["flair_tags"].apply(
    lambda x: ",".join([tag.text for tag in x if tag.tag == "PER"])
)

# extract the organization tags from the flair tags and add them to the dataframe as a comma separated string
df["flair_organizations"] = df["flair_tags"].apply(
    lambda x: ",".join([tag.text for tag in x if tag.tag == "ORG"])
)

# extract the misc tags from the flair tags and add them to the dataframe as a comma separated string
df["flair_misc"] = df["flair_tags"].apply(
    lambda x: ",".join([tag.text for tag in x if tag.tag == "MISC"])
)

# finally, drop the flair tags column
df.drop(columns=["flair_tags"], inplace=True)

# save the dataframe to a csv file
df.to_csv("22_10_22_hls_people_hist_hub_flair.csv", sep=";", index=False)
