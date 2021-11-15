from cleaning import get_filtered_df
from preprocessing import preprocess

if __name__ == "__main__":
    df = get_filtered_df()
    tokens_list, model = preprocess(df)