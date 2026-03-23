import pandas as pd
import warnings
import csv
import numpy as np
import langcodes
from datetime import datetime
import re
import os
import phonenumbers
import pycountry
from collections import Counter
from countryinfo import CountryInfo
import pycountry_convert
from countryinfo import CountryInfo, CountryNotFoundError
import seaborn as sns
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')


name_of_the_file = None

def is_list_of_lists(data):
    return isinstance(data, list) and all(isinstance(item, list) for item in data)


def write_to_file(filename):
    def decorator(func):
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            filename = name_of_the_file
            file_exists = os.path.isfile(filename)
            write_header = not file_exists or os.path.getsize(filename) == 0
            with open(filename, 'a', newline='') as f:
                csv_writer = csv.writer(f)
                if write_header:
                    csv_writer.writerow(["Check Name", "File Name", "Status", "DQ Message"])
                if result:
                    if is_list_of_lists(result):
                        csv_writer.writerows(result)
                    else:
                        csv_writer.writerow(result)
            return result
        return wrapper
    return decorator

@write_to_file(name_of_the_file)
def get_duplicates(df: pd.DataFrame, check_name: str, file_name: str) -> list:
    duplicates_cnt = len(df[df.duplicated(keep='first')])
    if duplicates_cnt > 0:
        return [check_name, file_name, "FAILED", f"Dataset contains '{duplicates_cnt}' full duplicates"]

@write_to_file(name_of_the_file)
def get_business_duplicates(df: pd.DataFrame, columns_to_check: list, check_name: str, file_name: str) -> list:
    duplicates_cnt = len(df[df.duplicated(subset=columns_to_check)])
    if duplicates_cnt > 0:
        return [check_name, file_name, "FAILED", f"Dataset contains '{duplicates_cnt}' duplicates by the columns: {columns_to_check}"]


@write_to_file(name_of_the_file)
def get_imbalances(df: pd.DataFrame, check_name: str, file_name: str, cat_threshold=0.3, skew_threshold=1.0) -> list:
    imbalances = []
    for col in df.columns:
        if df[col].dtype in ['object', 'category', 'str']:
            top_freq = df[col].value_counts(normalize=True).max()
            least_freq = df[col].value_counts(normalize=True).min()
            if top_freq > cat_threshold:
                dq_msg = f"Categorical imbalance for '{col}': Max value {top_freq:.2f}% of rows are '{df[col].value_counts().idxmax()}' and Min value {least_freq:.4f} of rows are '{df[col].value_counts().idxmin()}'" 
                imbalances.append([check_name, file_name, "FAILED", dq_msg])
        if df[col].dtype in ['number', 'int64', 'float64']:
            skew = df[col].skew()
            if abs(skew) > skew_threshold:
                dq_msg = f"Numerical imbalance for '{col}': skewness={skew:.2f}%"
                imbalances.append([check_name, file_name, "FAILED", dq_msg])
    return imbalances

@write_to_file(name_of_the_file)
def get_completeness(df: pd.DataFrame, check_name: str, file_name: str) -> list:
    completeness = []
    for col in df.columns:
        if pd.isna(df[col]).sum() > 0:
            completeness.append([check_name, file_name, "FAILED", f"'{col}' columns has {pd.isna(df[col]).sum()} missing or NaN/NaT values"])
    return completeness

@write_to_file(name_of_the_file)
def compare_dates_bias(df: pd.DataFrame, date1_col: str, date2_col: str, check_name: str, file_name: str) -> list:
    date1 = pd.to_datetime(df[date1_col], errors='coerce')
    date2 = pd.to_datetime(df[date2_col], errors='coerce')
    biased_dates = df[date1 < date2]
    if len(biased_dates) > 0:
        return [check_name, file_name, "FAILED", f"'{date1_col}' < '{date2_col}' for {len(biased_dates)} rows"]

@write_to_file(name_of_the_file)
def check_is_numeric(df: pd.DataFrame, check_name: str, file_name: str) -> list:
    for col in df.columns:
        numeric_col = pd.to_numeric(df[col], errors='coerce')
        if df[col].dtype in ['number', 'int64']:
            # is_all_numeric = numeric_col.notnull().all()
            non_numeric_values = df[col][numeric_col.isnull()]
            if len(non_numeric_values) > 0:
                return [check_name, file_name, "FAILED", f"'{col}' columns has not NUMERIC values {non_numeric_values.tolist()}" ]

@write_to_file(name_of_the_file)
def check_is_float(df: pd.DataFrame, check_name: str, file_name: str) -> list:
    for col in df.columns:
        if df[col].dtype in ['float', 'float64']:
            is_float = df[col].apply(lambda x: isinstance(x, float))
            non_float_values = df.loc[~is_float, col].tolist()
            if len(non_float_values) > 0:
                return [check_name, file_name, "FAILED", f"'{col}' columns has not FLOAT values {non_float_values}" ]

@write_to_file(name_of_the_file)
def check_non_ascii_char(df: pd.DataFrame, check_name: str, file_name: str) -> list:
    # skip_chars = ['¢', '²', '³', '½', 'ø', '⁴', '⅓', '☆']
    skip_chars = ['¡', '«', '·', '»', 'Æ', 'É', 'à', 'á', 'ã', 'ä', 'ç', 'è', 'é', 'í', 'ï', 'ñ', 'ò', 'ó', 'ô', 'ö', 'ú', 'û', 'ü', 'ā', 'ō', 'ū', '–', '—', '’', '…', '−']
    skip_chars_set = set(skip_chars)

    def is_valid_ascii_or_skipped(x):
        if not isinstance(x, str):
            return False
        return all(ch.isascii() or ch in skip_chars_set for ch in x)

    for col in df.select_dtypes(include=['object', 'category']):        
        mask_valid_ascii = df[col].apply(is_valid_ascii_or_skipped)
        non_ascii_rows = df[~mask_valid_ascii][col].dropna()
        if len(non_ascii_rows) > 0:
            non_ascii_chars = set()
            for val in non_ascii_rows:
                if isinstance(val, str):
                    non_ascii_chars.update([ch for ch in val if not ch.isascii()])
            return [check_name, file_name, "FAILED", f"'{col}' contains {len(non_ascii_rows)} rows with encoding issues. Non-ASCII characters: {sorted(non_ascii_chars)}"]


@write_to_file(name_of_the_file)
def check_description_length(df: pd.DataFrame, check_name: str, file_name: str) -> list:
    # Applicable only for movies.csv
    df_new = df.copy()
    df_new['has_two_words'] = df_new['description'].str.contains(r'\w+\s+\w+', na=False)
    short_desc_cnt = len(df_new.loc[df_new['has_two_words'] == False])
    return [check_name, file_name, "FAILED", f"Too short 'description' (less than two words) identified for {short_desc_cnt} rows"]


@write_to_file(name_of_the_file)
def check_date_format(df: pd.DataFrame, col_name: str, check_name: str, file_name: str) -> list:
    # Regex for YYYY-MM-DD format
    iso_date_regex = r'^\d{4}-\d{2}-\d{2}$'

    # Check if each string matches the regex
    is_valid_format = df[col_name].astype(str).str.match(iso_date_regex)
    num_invalid_format = (~is_valid_format).sum()

    if num_invalid_format > 0:
        return [check_name, file_name, "FAILED", f"'{col_name}' containst {num_invalid_format} rows with incorrect DATE format"]
    

@write_to_file(name_of_the_file)
def check_is_valid_isocode(df: pd.DataFrame, cols_to_check: list, check_name: str, file_name: str) -> list:
    invalid_isocodes = []
    for col_name in cols_to_check:
        is_valid = df[col_name].apply(lambda x: langcodes.tag_is_valid(x) if pd.notnull(x) else True)
        mask_invalid = (~is_valid) & df[col_name].notnull()
        invalid_values = df.loc[mask_invalid, col_name].tolist()
        if len(invalid_values) > 0:     
            invalid_isocodes.append([check_name, file_name, "FAILED", f"'{col_name}' containst {len(invalid_values)} rows with incorrect ISO language codes {list(set(invalid_values))}"])
    return invalid_isocodes

@write_to_file(name_of_the_file)
def check_is_one_of_acceptable(df: pd.DataFrame, col_name: str, acceptable_values: list, check_name: str, file_name: str) -> list:
    is_acceptable = df[col_name].isin(acceptable_values)
    unacceptable_values = df.loc[~is_acceptable, col_name].unique().tolist()
    if len(unacceptable_values) > 0:
        return [check_name, file_name, "FAILED", f"'{col_name}' contains {len(unacceptable_values)} incorrect values - {unacceptable_values}"]
    
@write_to_file(name_of_the_file)
def check_no_future_dates(df: pd.DataFrame, col_name: str, check_name: str, file_name: str) -> list:
    dates = pd.to_datetime(df[col_name], errors='coerce')
    today = pd.Timestamp(datetime.today().date())
    future_dates = df.loc[dates > today, col_name]
    if len(future_dates.tolist()) > 0:
        return [check_name, file_name, "FAILED", f"Column '{col_name}' contains {len(future_dates.tolist())} future dates - {future_dates.tolist()}"] 

@write_to_file(name_of_the_file)
def check_email_format(df: pd.DataFrame, col_name: str, check_name: str, file_name: str) -> list:
    # Rgex for email validation
    email_regex = r'^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$'
    is_valid_email = df[col_name].apply(lambda x: bool(re.match(email_regex, str(x))) if pd.notnull(x) else False)
    invalid_emails = df.loc[~is_valid_email, col_name]
    if len(invalid_emails.tolist()) > 0:
        return [check_name, file_name, "FAILED", f"Column '{col_name}' contains {len(invalid_emails.tolist())} future dates - {invalid_emails.tolist()}"] 
    
@write_to_file(name_of_the_file)
def check_valid_country_name(df: pd.DataFrame, country_col: set, check_name: str, file_name: str) -> list:
    valid_countries = set()
    for country in pycountry.countries:
        valid_countries.add(country.name.lower())
        valid_countries.add(getattr(country, 'official_name', '').lower())
    valid_countries.discard('')  # Remove empty string if present

    invalid_entries = []
    for idx, row in df.iterrows():
        user_id = row['user_id']
        country = row[country_col]
        if pd.isnull(country):
            invalid_entries.append([check_name, file_name, "FAILED", f"User with id '{user_id}' contains an empty country"])
        if str(country).strip().lower() not in valid_countries:
            invalid_entries.append([check_name, file_name, "FAILED", f"User with id '{user_id}' contains an incorrect country '{country}'"])
    if len(invalid_entries) > 0:
        return [check_name, file_name, "FAILED", f"File contains {len(invalid_entries)} incorrect country names in '{country_col}' column"]

@write_to_file(name_of_the_file)
def check_phone_by_country(df: pd.DataFrame, phone_col: str, country_col: str, check_name: str, file_name: str) -> list:
    invalid_entries = []
    country_phone_counter = Counter()

    for idx, row in df.iterrows():
        user_id = row['user_id']
        phone = row[phone_col]
        country = row[country_col]
        if pd.isnull(phone) or pd.isnull(country):
            # invalid_entries.append([check_name, file_name, "FAILED", f"User with id '{user_id}' contains an empty phone or country"])
            pass
        try:
            parsed = phonenumbers.parse(str(phone), str(country))
            if not phonenumbers.is_valid_number(parsed):
                # invalid_entries.append([check_name, file_name, "FAILED", f"User with id '{user_id}' contains an incorrect phone {phone} for his country {country}"])
                country_phone_counter[country] += 1
        except phonenumbers.NumberParseException:
            # invalid_entries.append([check_name, file_name, "FAILED", f"User with id '{user_id}' contains an incorrect phone {phone} for his country {country}"])
            country_phone_counter[country] += 1
    if len(country_phone_counter) > 0:
        for country, count in country_phone_counter.items():
            invalid_entries.append([check_name, file_name, "FAILED", f"Country '{country}' has {count} incorrect phone numbers"])
    return invalid_entries
    
@write_to_file(name_of_the_file)
def check_city_by_country(input_df: pd.DataFrame, lookup_df: pd.DataFrame, city_col: str, country_col: str, check_name: str, file_name: str) -> list:
    invalid_entries = []
    country_city_counter = Counter()

    # merge input df with the lookup df (it needs to have predefined column names)
    df = pd.merge(input_df, lookup_df, left_on=[country_col, city_col], right_on=['country_name', 'city_name'], how="left")

    for idx, row in df.iterrows():
        user_id = row['user_id']
        city = row[city_col]
        country = row[country_col]
        lookup_city = row['city_name']
        if pd.isnull(lookup_city):
            # invalid_entries.append([check_name, file_name, "FAILED", f"User with id '{user_id}' contains an incorrect city '{city}' or country '{country}'"])
            country_city_counter[country] += 1

    if len(country_city_counter) > 0:
        for country, count in country_city_counter.items():
            invalid_entries.append([check_name, file_name, "FAILED", f"Country '{country}' has {count} incorrect city(ies)"])
    return invalid_entries

def language_name_to_code(language_name):
    try:
        lang = pycountry.languages.lookup(language_name)
        if hasattr(lang, 'alpha_2'):
            return lang.alpha_2.lower()
        elif hasattr(lang, 'alpha_3'):
            return lang.alpha_3.lower()
    except LookupError:
        return None

@write_to_file(name_of_the_file)
def check_language_by_country(df: pd.DataFrame, language_col: str, country_col: str, check_name: str, file_name: str) -> list:
    invalid_entries = []
    country_counter = Counter()

    for idx, row in df.iterrows():
        user_id = row['user_id']
        language = row[language_col]
        country = row[country_col]
        if pd.isnull(language) or pd.isnull(country):
            # invalid_entries.append([check_name, file_name, "FAILED", f"User with id '{user_id}' contains an empty language or country"])
            continue
        lang_code = language_name_to_code(str(language).strip())
        try:
            info = CountryInfo(str(country).strip())
            official_languages = info.languages()
            official_languages_normalized = [code.lower().strip() for code in official_languages]
            if lang_code not in official_languages_normalized:
                # invalid_entries.append([check_name, file_name, "FAILED", f"User with id '{user_id}' contains an incorrect language {language} for his country {country}"])
                country_counter[country] += 1
        except CountryNotFoundError:
            # invalid_entries.append([check_name, file_name, "FAILED", f"User with id '{user_id}' contains an incorrect country {country}, not found in CountryInfo"])
            country_counter[country] += 1

    if len(country_counter) > 0:
        for country, count in country_counter.items():
            invalid_entries.append([check_name, file_name, "FAILED", f"Country '{country}' has {count} incorrect languages"])
    return invalid_entries

@write_to_file(name_of_the_file)
def check_consistency(df_1: pd.DataFrame, df_2: pd.DataFrame, join_column: str, check_column: str, check_name: str, file_name: str) -> list:
    na_df = pd.merge(df_1, df_2, on=[join_column], how="left")
    na_cnt = len(na_df.loc[pd.isnull(na_df[check_column])])
    if na_cnt > 0:
        return [check_name, file_name, "FAILED", f"Non existing {na_cnt} rows in {join_column}"] 
    
@write_to_file(name_of_the_file)
def check_uneven_distribution(df: pd.DataFrame, group_column: str, check_name: str, file_name: str) -> list:
    group_counts = df[group_column].value_counts(dropna=False)
    total = len(df)
    group_percent = group_counts / total * 100
    result = pd.DataFrame({
        'group': group_counts.index,
        'count': group_counts.values,
        'percentage': group_percent.round(2).values
    })
    if len(result) > 0:
        return[check_name, file_name, "FAILED", f"Uneven values distribution for column '{group_column}' - {result[['group', 'percentage']].values.tolist()}"]

def categorize_by_generation(df: pd.DataFrame, categorize_col: str, age_groups: dict, output_col='generation') -> pd.DataFrame:
    df[categorize_col] = pd.to_datetime(df[categorize_col], errors='coerce')
    df['birth_year'] = df[categorize_col].dt.year

    def get_generation(year):
        if pd.isnull(year):
            return None
        for group in age_groups:
            if group["start"] <= year <= group["end"]:
                return group["name"]

    df[output_col] = df['birth_year'].apply(get_generation)
    df.drop(columns=['birth_year'], inplace=True)
    return df

def categorize_by_rating_groups(df: pd.DataFrame, categorize_col: str, rating_groups: dict, output_col='rating_group') -> pd.DataFrame:
    def get_generation(rating):
        if pd.isnull(rating):
            return None
        for group in rating_groups:
            if group["start"] <= rating < group["end"]:
                return group["name"]

    df[output_col] = df[categorize_col].apply(get_generation)
    return df

def country_to_continent(country_name):
    try:
        country = pycountry.countries.lookup(country_name)
        country_code = country.alpha_2
        continent_code = pycountry_convert.country_alpha2_to_continent_code(country_code)
        continent_name = pycountry_convert.convert_continent_code_to_continent_name(continent_code)
        return continent_name
    except Exception:
        return country_name

def categorize_by_continent(df, country_col, output_col='continent'):
    df[output_col] = df[country_col].apply(lambda x: country_to_continent(x) if pd.notnull(x) else None)
    return df

def check_imbalances(df: pd.DataFrame, group_column: str) -> list:
    group_counts = df[group_column].value_counts(dropna=False)
    total = len(df)
    group_percent = group_counts / total * 100
    result = pd.DataFrame({
        'group': group_counts.index,
        'count': group_counts.values,
        'percentage': group_percent.round(2).values
    })
    return result

def extract_adjectives_verbs_by_rating(df: pd.DataFrame, review_col: str, rating_col: str, adjectives_txt_path: str, verbs_txt_path: str, stop_words: set):
    with open(adjectives_txt_path, 'r', encoding='utf-8') as f:
        adjectives_set = set(line.strip().lower() for line in f if line.strip())
    with open(verbs_txt_path, 'r', encoding='utf-8') as f:
        verbs_set = set()
        for line in f:
            verbs = [v.strip().lower() for v in line.strip().split(',') if v.strip()]
            verbs_set.update(verbs)
    
    positive_words, neutral_words, negative_words = [], [], []
    
    for _, row in df.iterrows():
        review = str(row[review_col])
        try:
            rating = float(row[rating_col])
        except (ValueError, TypeError):
            continue  # skip rows with invalid rating
        words = re.findall(r'\b\w+\b', review.lower())
        filtered_words = [
            w for w in words 
            if w not in stop_words and len(w) > 1 and (w in adjectives_set or w in verbs_set)
        ]
        if rating >= 4.0:
            positive_words.extend(filtered_words)
        elif 3.0 <= rating < 4.0:
            neutral_words.extend(filtered_words)
        elif rating < 3.0:
            negative_words.extend(filtered_words)
    
    return (
        list(set(positive_words)),
        list(set(negative_words)),
        list(set(neutral_words))
    )

def add_sentiment_counts_column(df, text_col, positive_list, negative_list, neutral_list, new_col='sentiment_counts'):
    positive_set = set(w.lower() for w in positive_list)
    negative_set = set(w.lower() for w in negative_list)
    neutral_set  = set(w.lower() for w in neutral_list)
    
    def count_sentiment_words(text):
        words = re.findall(r'\b\w+\b', str(text).lower())
        positive_count = sum(1 for w in words if w in positive_set)
        negative_count = sum(1 for w in words if w in negative_set)
        neutral_count  = sum(1 for w in words if w in neutral_set)
        return {
            "positive": positive_count,
            "negative": negative_count,
            "neutral": neutral_count
        }
    
    df[new_col] = df[text_col].apply(count_sentiment_words)
    return df


def add_expected_sentiment(df, categories_col='movie_word_categories', new_col='expected_sentiment'):
    def classify_sentiment(cat):
        pos = cat.get('positive', 0)
        neg = cat.get('negative', 0)
        neu = cat.get('neutral', 0)
        # Find which is dominant
        counts = {'positive': pos, 'negative': neg, 'neutral': neu}
        max_count = max(counts.values())
        dominant = [k for k, v in counts.items() if v == max_count]
        # Case 4: No dominant category
        if len(dominant) != 1:
            return 'neutral'
        dom = dominant[0]
        # Case 1: Positive is dominant
        if dom == 'positive':
            if neu - 1 + neg < pos:
                return 'positive'
            else:
                return 'neutral'
        # Case 2: Neutral is dominant
        elif dom == 'neutral':
            return 'neutral'
        # Case 3: Negative is dominant
        elif dom == 'negative':
            if pos + neu >= neg:
                return 'neutral'
            else:
                return 'negative'
    
    df[new_col] = df[categories_col].apply(classify_sentiment)
    return df