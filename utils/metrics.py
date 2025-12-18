import pandas as pd
import numpy as np

def clean_delivery_data(df):
    df = df.copy()

    df["delay_minutes"] = df["actual_delivery_time"] - df["promised_delivery_time"]
    df["delayed"] = df["delay_minutes"].apply(lambda x: 1 if x > 0 else 0)

    return df
