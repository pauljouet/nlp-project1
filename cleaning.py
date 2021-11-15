import os
import pandas as pd
from langdetect import detect

RELEVANT_COLUMNS = ['code', 'product_name', 'ingredients_text', 'packaging_text']
DIR = os.path.dirname(__file__)
FILT_DATA_PATH = os.path.join(DIR, "data/filtered_openfoodfacts.csv")
DATA_PATH = os.path.join(DIR, "data/openfoodfacts.csv")

def country_filter(df) -> pd.DataFrame:
    """Filters products which do not exist in France"""
    return df.loc[df['countries_tags'].str.contains("france", na=False)]

def na_filter(df) -> pd.DataFrame:
    """Filters products without an ingredient list"""
    return df.dropna(subset= ['ingredients_text', 'product_name']).filter(items= RELEVANT_COLUMNS)

# ~45 minutes on 250k entries
def try_detect(text: str) -> str:
    """Returns the language of the text. 'fr' for French"""
    try:
        language = detect(text)
    except:
        language = "error"
        # print("This throws an error:", text)
    return language

def language_filter(df) -> pd.DataFrame:
    """Filters products whose ingredient list is not in French"""
    df['detected_lang'] = df['ingredients_text'].apply(try_detect)
    return df.loc[df['detected_lang'].str.contains("fr", na=False)]

def filter_df(df) -> pd.DataFrame:
    """Applies all filters on the DataFrame"""
    df1 = country_filter(df)
    df2 = na_filter(df1)
    return language_filter(df2)

def get_filtered_df() -> pd.DataFrame:
    """Uses the database contained in the data folder, or downloads it to create a DataFrame, filter it, and write it into a new .csv file"""
    if os.path.exists(FILT_DATA_PATH):
        return pd.read_csv(FILT_DATA_PATH, encoding= 'utf-8', delimiter= '\t')
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH, encoding = 'utf-8', delimiter="\t")
    else:
        df = pd.read_csv("https://static.openfoodfacts.org/data/en.openfoodfacts.org.products.csv", encoding = 'utf-8', delimiter="\t")
    res = filter_df(df)
    res.to_csv(FILT_DATA_PATH, sep= "\t")
    return res

if __name__ == "__main__":
    get_filtered_df()