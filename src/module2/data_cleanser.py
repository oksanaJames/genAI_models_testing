import pandas as pd
import numpy as np
from faker import Faker
import datetime
import pycountry
import re


def remove_duplicates(df: pd.DataFrame, subset_columns: list) -> pd.DataFrame:
    df_cleaned = df.drop_duplicates(subset=subset_columns)
    return df_cleaned

def remove_na_rows_by_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    all_na = df[columns].isna().all(axis=1)
    return df.loc[~all_na]

def replace_na_by_column(df: pd.DataFrame, notna_column: str, na_column: str) -> pd.DataFrame:
    df[na_column] = np.where(pd.isna(df[na_column]), df[notna_column], df[na_column])
    return df

def replace_column_value(df: pd.DataFrame, column_name: str, filter_value: str, replace_with: str) -> pd.DataFrame:
    df[column_name] = np.where(df[column_name] == filter_value, replace_with, df[column_name])
    return df

def convert_date_format(df: pd.DataFrame, column: str) -> pd.DataFrame:
    df[column] = pd.to_datetime(df[column], format='%m/%d/%Y', errors='coerce').dt.strftime('%Y-%m-%d')
    return df[~pd.isna(df[column])]

def remove_spec_chars(df: pd.DataFrame, column: str, chars_to_remove: list) -> pd.DataFrame:
    pattern = '[' + ''.join('\\' + c for c in chars_to_remove) + ']'
    df[column] = df[column].astype(str).str.replace(pattern, '', regex=True)
    return df

def generate_synthetic_movies_data(df: pd.DataFrame, language_sound: list, original_language: list, 
                                   mpa_film_rating: list, start_date: str, end_date: str, 
                                   n_rows=1000) -> pd.DataFrame:
    fake = Faker()
    schema = df.columns.tolist()
    start_date = datetime.datetime.strptime(start_date, "%Y-%m-%d").date()
    end_date = datetime.datetime.strptime(end_date, "%Y-%m-%d").date()
    
    data = []
    for i in range(n_rows):
        row = {}
        row['movie_id'] = 9838 + i
        row['title'] = fake.sentence(nb_words=3).rstrip('.')
        row['description'] = fake.text(max_nb_chars=200)
        row['release_date'] = fake.date_between(start_date, end_date)
        row['language_sound'] = np.random.choice(language_sound)
        row['Original_Language'] = np.random.choice(original_language)
        row['MPA_film_rating'] = np.random.choice(mpa_film_rating)
        data.append(row)
    
    synthetic_df = pd.DataFrame(data, columns=schema)
    synthetic_movie_ids = synthetic_df['movie_id'].to_list()
    combined_df = pd.concat([df, synthetic_df], ignore_index=True)
    return combined_df, synthetic_movie_ids

def generate_synthetic_users_data(
    df: pd.DataFrame,
    n_rows: int = 1000,
    user_id_start: int = 5001,
    dob_start_year: int = 1981,
    dob_end_year: int = 1996,
    email_regex: str = r'^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$'
) -> pd.DataFrame:
    fake = Faker()
    schema = df.columns.tolist()
    
    # Get all countries except 'Russian Federation'
    countries = [country.name for country in pycountry.countries if country.name != 'Russian Federation']
    
    # Map country to language using pycountry
    country_languages = {}
    for country in pycountry.countries:
        if country.name == 'Russian Federation':
            continue
        langs = []
        try:
            country_obj = pycountry.countries.get(name=country.name)
            if hasattr(country_obj, 'alpha_2'):
                langs = [lang.name for lang in pycountry.languages if hasattr(lang, 'alpha_2')]
        except Exception:
            pass
        country_languages[country.name] = langs if langs else [fake.language_name()]
    
    data = []
    for i in range(n_rows):
        row = {}
        row['user_id'] = user_id_start + i
        row['first_name'] = fake.first_name()
        row['last_name'] = fake.last_name()
        start_date = datetime.date(dob_start_year, 1, 1)
        end_date = datetime.date(dob_end_year, 12, 31)
        row['date_of_birth'] = fake.date_between(start_date, end_date).strftime('%Y-%m-%d')
        email = fake.email()
        while not pd.Series([email]).str.match(email_regex).iloc[0]:
            email = fake.email()
        row['email'] = email
        country = np.random.choice(countries)
        row['country'] = country
        try:
            country_obj = pycountry.countries.get(name=country)
            locale = country_obj.alpha_2.lower()
            fake_local = Faker(locale)
        except Exception:
            fake_local = fake
        try:
            row['city'] = fake_local.city()
        except AttributeError:
            row['city'] = fake.city()
        try:
            row['contact_number'] = fake_local.phone_number()
        except AttributeError:
            row['contact_number'] = fake.phone_number()
        langs = country_languages.get(country, [fake.language_name()])
        row['language'] = np.random.choice(langs)
        row['is_active'] = 1.0
        data.append(row)
    
    synthetic_df = pd.DataFrame(data, columns=schema)
    synthetic_user_ids = synthetic_df['user_id'].to_list()
    combined_df = pd.concat([df, synthetic_df], ignore_index=True)
    return combined_df, synthetic_user_ids

def generate_synthetic_reviews_data(
    df: pd.DataFrame,
    user_ids: list,
    movie_ids: list,
    review_id_start: int = 15001,
    rating_min: float = 0.0,
    rating_max: float = 5.0
) -> pd.DataFrame:
    fake = Faker()
    schema = df.columns.tolist()
    
    positive_phrases = [
       "fantastic", "amazing", "incredible", "outstanding", "brilliant", "superb", "excellent", "wonderful"
    ]
    negative_phrases = [
        "disappointing", "boring", "terrible", "poor", "bad", "awful", "mediocre", "waste of time"
    ]
    neutral_phrases = [
        "okay", "an average", "decent", "could be better", "nothing special", "fine", "acceptable"
    ]
    
    data = []
    for i, user_id in enumerate(user_ids):
        row = {}
        row['review_id'] = review_id_start + i
        row['user_id'] = user_id
        row['movie_id'] = np.random.choice(movie_ids)
        rating = round(np.random.uniform(rating_min, rating_max), 1)
        row['rating'] = rating
        if rating >= 4.0:
            phrase = np.random.choice(positive_phrases)
            review_text = f"This movie is {phrase}. I reccomend it"
        elif rating <= 2.0:
            phrase = np.random.choice(negative_phrases)
            review_text = f"This movie is {phrase}. I don't recommend it"
        else:
            phrase = np.random.choice(neutral_phrases)
            review_text = f"This movie is {phrase}"
        row['review_text'] = review_text
        row['review_date'] = ""
        data.append(row)
    
    synthetic_df = pd.DataFrame(data, columns=schema)
    combined_df = pd.concat([df, synthetic_df], ignore_index=True)
    return combined_df

