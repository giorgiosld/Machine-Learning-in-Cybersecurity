from preprocess import preprocess_data
import pandas as pd


sorted_word_counts = preprocess_data('train-mails')
df = pd.DataFrame.from_dict(sorted_word_counts, orient='index', columns=['count'])
print(df.head(2000))
