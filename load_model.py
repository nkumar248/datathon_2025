import joblib
import pandas as pd
import numpy as np
from rapidfuzz.distance import JaroWinkler
import pycountry
from datetime import datetime

def load_model_and_features():
    """Load the trained model and feature columns"""
    model = joblib.load('trained_model.joblib')
    feature_columns = joblib.load('feature_columns.joblib')
    return model, feature_columns

def load_and_preprocess_data(file_path):
    """
    Load and preprocess the data from a JSONL file
    """
    # Load data
    df = pd.read_json(file_path, lines=True)
    
    # Validate country codes
    df["correct_country_code"] = True
    for i, row in df.iterrows():
        try:
            country = pycountry.countries.get(name=row["passport_country"])
            df.loc[i, "correct_country_code"] = (row["passport_country_code"] == country.alpha_3)
        except:
            df.loc[i, "correct_country_code"] = False

    # Validate MRZ codes
    df["mrz_0_false"] = False
    df["mrz_1_false"] = False
    for i in df.index:
        # Validate first MRZ line
        mrz1 = df.loc[i, "passport_passport_mrz_0"][:5] + df.loc[i, "passport_last_name"].upper() + "<<" + df.loc[i, "passport_first_name"].upper()
        if df.loc[i, "passport_middle_name"] != "":
            mrz1 += "<" + df.loc[i, "passport_middle_name"].upper()
        mrz1 += "<" * (45 - len(mrz1))
        df.loc[i, "mrz_0_false"] = (mrz1 != df.loc[i, "passport_passport_mrz_0"])
        
        # Validate second MRZ line
        mrz2 = df.loc[i, "passport_passport_number"] + df.loc[i, "passport_passport_mrz_0"][2:5]
        bday = df.loc[i, "passport_birth_date"].split("-")
        bday = bday[0][2:] + bday[1] + bday[2]
        mrz2 += bday
        mrz2 += "<" * (45 - len(mrz2))
        df.loc[i, "mrz_1_false"] = (mrz2 != df.loc[i, "passport_passport_mrz_1"])

    # Check for zero salaries
    df["salary_zero"] = False
    for i in range(5):
        df.loc[df[f"client_profile_employment_history_{i}_salary"] == 0, "salary_zero"] = True

    # Combine name fields
    df = combine_name_fields(df)
    
    return df

def combine_name_fields(df):
    """
    Combine and clean name fields from different sources
    """
    # Passport names
    df['passport_full_name'] = df['passport_first_name'].fillna('') + ' ' + \
                              df['passport_middle_name'].fillna('') + ' ' + \
                              df['passport_last_name'].fillna('')
    df['passport_full_name'] = df['passport_full_name'].str.strip().replace('\s+', ' ', regex=True)
    df = df.drop(['passport_first_name', 'passport_middle_name', 'passport_last_name'], axis=1)

    # Account form names
    df['account_form_full_name'] = df['account_form_first_name'].fillna('') + ' ' + \
                                  df['account_form_middle_name'].fillna('') + ' ' + \
                                  df['account_form_last_name'].fillna('')
    df['account_form_full_name'] = df['account_form_full_name'].str.strip().replace('\s+', ' ', regex=True)
    df = df.drop(['account_form_first_name', 'account_form_middle_name', 'account_form_last_name'], axis=1)

    # Client profile names
    df['client_profile_full_name'] = df['client_profile_name'].fillna('')
    df['client_profile_full_name'] = df['client_profile_full_name'].str.strip().replace('\s+', ' ', regex=True)
    df = df.drop(['client_profile_name'], axis=1)
    
    return df

def calculate_similarities(df):
    """
    Calculate similarities between corresponding fields
    """
    # Define column pairs to compare
    column_pairs = {
    "passport_full_name" : ["client_profile_full_name", "account_form_full_name", "llm_full_name"],
    "passport_gender" : ["client_profile_gender", "llm_gender"],
    "passport_country" : ["client_profile_nationality", "llm_nationality", "passport_nationality"],
    "passport_nationality" : ["client_profile_nationality", "llm_nationality"],
    "passport_birth_date" : ["client_profile_birth_date"],
    "passport_passport_number" : ["client_profile_passport_number", "account_form_passport_number"],
    "passport_passport_expiry_date" : ["client_profile_passport_expiry_date"],
    "passport_passport_issue_date" : ["client_profile_passport_issue_date"],
    "client_profile_address_city" : ["account_form_address_city"],
    "client_profile_address_street_name" : ["account_form_address_street_name"],
    "client_profile_address_street_number" : ["account_form_address_street_number"],
    "client_profile_address_postal_code" : ["account_form_address_postal_code"],
    "client_profile_country_of_domicile" : ["account_form_country_of_domicile", "llm_country_of_residence"],
    "client_profile_phone_number" : ["account_form_phone_number"],
    "client_profile_email_address" : ["account_form_email_address"],
    "client_profile_marital_status" : ["llm_marital_status"],
    "account_form_full_name" : ["account_form_name", "llm_full_name"],
    "client_profile_secondary_school_name" : ["llm_secondary_school_name"],
    "client_profile_secondary_school_graduation_year" : ["llm_secondary_school_graduation_year"],
    "client_profile_higher_education_0_university" : ["llm_university1_name"],
    "client_profile_higher_education_1_university" : ["llm_university2_name"], 
    "client_profile_higher_education_0_graduation_year" : ["llm_university1_graduation_year"],
    "client_profile_higher_education_1_graduation_year" : ["llm_university2_graduation_year"],
    "client_profile_employment_history_0_start_year" : ["llm_employment1_start_year"],
    "client_profile_employment_history_0_end_year" : ["llm_employment1_end_year"],
    "client_profile_employment_history_0_company" : ["llm_employment1_company"],
    "client_profile_employment_history_0_position" : ["llm_employment1_position"],
    "client_profile_employment_history_0_salary" : ["llm_employment1_salary"],
    "client_profile_employment_history_1_start_year" : ["llm_employment2_start_year"],
    "client_profile_employment_history_1_end_year" : ["llm_employment2_end_year"],
    "client_profile_employment_history_1_company" : ["llm_employment2_company"],
    "client_profile_employment_history_1_position" : ["llm_employment2_position"],
    "client_profile_employment_history_1_salary" : ["llm_employment2_salary"],
    "client_profile_employment_history_2_start_year" : ["llm_employment3_start_year"],
    "client_profile_employment_history_2_end_year" : ["llm_employment3_end_year"],
    "client_profile_employment_history_2_company" : ["llm_employment3_company"],
    "client_profile_employment_history_2_position" : ["llm_employment3_position"],
    "client_profile_employment_history_2_salary" : ["llm_employment3_salary"],
    "client_profile_employment_history_3_start_year" : ["llm_employment4_start_year"],
    "client_profile_employment_history_3_end_year" : ["llm_employment4_end_year"],
    "client_profile_employment_history_3_company" : ["llm_employment4_company"],
    "client_profile_employment_history_3_position" : ["llm_employment4_position"],
    "client_profile_employment_history_3_salary" : ["llm_employment4_salary"],
    "client_profile_aum_savings" : ["llm_aum_savings"],
    "client_profile_aum_inheritance" : ["llm_aum_inheritance"],
    "client_profile_inheritance_details_profession" : ["llm_inheritance_details_profession"],
    "client_profile_inheritance_details_inheritance_year" : ["llm_inheritance_details_inheritance_year"],
    "client_profile_aum_real_estate_value" : ["llm_aum_real_estate_value"],
    "client_profile_real_estate_details_0_property_type" : ["llm_property1_type"],
    "client_profile_real_estate_details_0_property_value" : ["llm_property1_value"],
    "client_profile_real_estate_details_0_property_location" : ["llm_property1_location"],
    "client_profile_real_estate_details_1_property_type" : ["llm_property2_type"],
    "client_profile_real_estate_details_1_property_value" : ["llm_property2_value"],
    "client_profile_real_estate_details_1_property_location" : ["llm_property2_location"],
    "client_profile_real_estate_details_2_property_type" : ["llm_property3_type"],
    "client_profile_real_estate_details_2_property_value" : ["llm_property3_value"],
    "client_profile_real_estate_details_2_property_location" : ["llm_property3_location"],
    "client_profile_currency" : ["account_form_currency"],
    "client_profile_passport_number" : ["account_form_passport_number"],
    "client_profile_nationality" : ["llm_nationality"],
    "account_form_name" : ["llm_full_name"]}

    similarity_rows = []
    for idx, row in df.iterrows():
        client_id = row['client_id']
        
        similarities = {}
        for col1, col2_list in column_pairs.items():
            for col2 in col2_list:
                str1 = str(row[col1]) if pd.notna(row[col1]) else ''
                str2 = str(row[col2]) if pd.notna(row[col2]) else ''
                
                if len(str1) == 0 and len(str2) == 0:
                    similarity = 1.0
                else:
                    similarity = JaroWinkler.normalized_similarity(str1, str2)
                
                similarities[f"{col1}_{col2}"] = similarity
        
        similarities['client_id'] = client_id
        similarity_rows.append(similarities)

    similarities_df = pd.DataFrame(similarity_rows)
    
    # Add additional features
    similarities_df["expired_passport"] = (pd.to_datetime(df.passport_passport_expiry_date) < pd.Timestamp("2025-10-01")).astype(int)
    similarities_df["mrz_0_false"] = df.mrz_0_false.astype(int)
    similarities_df["mrz_1_false"] = df.mrz_1_false.astype(int)
    similarities_df["salary_zero"] = df.salary_zero.astype(int)
    similarities_df["correct_country_code"] = df.correct_country_code.astype(int)
    
    return similarities_df

def predict_on_new_data(file_path):
    """Load model and make predictions on new data"""
    # Load model and features
    model, feature_columns = load_model_and_features()
    
    # Process new data
    df = load_and_preprocess_data(file_path)
    similarities_df = calculate_similarities(df)
    
    # Make predictions
    X = similarities_df.drop(['client_id'], axis=1)
    predictions = model.predict(X[feature_columns])
    
    # Create results dataframe with just client_id and prediction
    results_df = pd.DataFrame({
        'client_id': similarities_df['client_id'],
        'prediction': ['Accept' if pred == 1 else 'Reject' for pred in predictions]
    })

    results_df = pd.DataFrame({
    'client_id': ['client_' + str(cid) for cid in similarities_df['client_id']],
    'prediction': ['Accept' if pred == 1 else 'Reject' for pred in predictions]
})
    
    # Save to CSV
    results_df.to_csv('predictions.csv', index=False, header=False, sep=';')

    return results_df

if __name__ == "__main__":
    # Example usage
    new_data_path = '/Users/saji/Desktop/DT_JB/Data/EVA_3.jsonl'
    results = predict_on_new_data(new_data_path)
    print(f"Predictions saved to predictions.csv")
    print(f"Total predictions: {len(results)}")
    #print(f"Accepted applications: {sum(results['prediction'])}")
    #print(f"Rejected applications: {len(results) - sum(results['prediction'])}")
