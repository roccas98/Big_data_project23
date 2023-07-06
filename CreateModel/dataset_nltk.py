import pandas as pd
import numpy as np
from pre_processing_nltk import denoise_text
import time
import sys

# Load the dataset
col_names = ["target", "ids", "date", "flag", "user", "text"]
current_path = sys.path[0]
filename = current_path + "/tweet_data.csv"
df = pd.read_csv(filename,encoding='latin-1', names=col_names)

# Pick 100000 random samples
df=df.sample(50000)

# Replace the 4 with 1 
df['target']=df['target'].replace(4,1)

# We pick up only the text and the target, containing the label and the tweet text
data = df['text']
labels = np.array(df['target'])

# We apply the denoise_text function to the text of the tweets to clean them
data = data.apply(denoise_text)
data = data.dropna()

# Function that returns the dataset already preprocessed
def get_data():
    return data, labels



