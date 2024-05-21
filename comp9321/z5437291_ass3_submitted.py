import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.metrics import (
    classification_report,
    precision_score,
    f1_score,
    mean_squared_error,
)
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.utils import shuffle
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    BaggingClassifier,
    StackingClassifier,
    VotingClassifier,
    RandomForestRegressor,
    StackingRegressor,
    GradientBoostingRegressor,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
import re
import sys


def main(train_file, test_file):
    # Read datasets
    df_train = pd.read_csv(train_file)
    df_test = pd.read_csv(test_file)

    df_train_policyid = df_train["policy_id"]
    df_test_policyid = df_test["policy_id"]

    # function to summarise the dataset before preprocessing
    def data_summary(dataset):
        print("Data Summary for", dataset)
        try:
            df = dataset
            print("Number of rows:", df.shape[0])
            print("Number of columns:", df.shape[1])
        except Exception as e:
            print("Error occurred while accessing DataFrame shape:", e)
        print("++++++++++++++++++++++++++++++++++++++++++++++")
        print("\nColumn Names:")
        print([str(col) for col in df.columns.tolist()])
        print("++++++++++++++++++++++++++++++++++++++++++++++")
        print("\nMissing Values:")
        print(df.isnull().sum())
        print("++++++++++++++++++++++++++++++++++++++++++++++")
        total_claims = (df["is_claim"] == 1).sum()
        total_no_claims = (df["is_claim"] == 0).sum()

        print(f"\nTotal Number of claims lodged: {total_claims}")
        print(f"Total Number of No claims lodged: {total_no_claims}\n")

    # preprocessing the existing data
    def convert_age_to_float(age_str):
        # Regular expression to extract years and months
        pattern = r"(\d+) years? and (\d+) months?|(\d+) years?|(\d+) months?"

        # Extract years and months from the string
        matches = re.search(pattern, age_str)
        if matches:
            if matches.group(1) and matches.group(2):
                years = int(matches.group(1))
                months = int(matches.group(2))
            elif matches.group(3):
                years = int(matches.group(3))
                months = 0
            elif matches.group(4):
                years = 0
                months = int(matches.group(4))

            return years + months / 12
        else:
            return np.nan

    def extract_number(area_str):
        # Regular expression to extract the number
        pattern = r"(\d+)"

        # Extract the number from the string
        match = re.search(pattern, area_str)
        if match:
            return int(match.group(1))
        else:
            return np.nan

    def convert_fuel_type_to_number(fuel_type):
        fuel_type_mapping = {"Petrol": 1, "Diesel": 2, "CNG": 3}
        return fuel_type_mapping.get(fuel_type, np.nan)

    def convert_transmission_type_to_number(transmission_type):
        transmission_type_mapping = {"Automatic": 1, "Manual": 2}
        return transmission_type_mapping.get(transmission_type, np.nan)

    def convert_steering_type_to_number(steering_type):
        steering_type_mapping = {"Power": 1, "Electric": 2, "Manual": 3}
        return steering_type_mapping.get(steering_type, np.nan)

    def convert_segment_to_number(segment):
        segment_mapping = {"A": 1, "B1": 2, "B2": 2, "C1": 3, "C2": 3, "Utility": 4}
        return segment_mapping.get(segment, np.nan)

    def dataCleanupWrangling(df):
        for column in df.columns:
            if column in [
                "Unnamed: 0",
                "policy_id",
            ]:  # Removed "Unnamed" and "policy_id"
                df.drop(column, axis=1, inplace=True)
            elif "is_" in column and df[column].dtype == "object":
                df[column] = (df[column].str.lower() == "yes").astype(int)
            elif column == "max_torque" and df[column].dtype == "object":
                torque_rpm = df[column].str.extract(r"(\d+\.*\d*)Nm@(\d+)rpm")
                df["max_torque_nm"] = torque_rpm[0].astype(float)
                df["max_torque_rpm"] = torque_rpm[1].astype(int)
                df.drop(column, axis=1, inplace=True)
            elif column == "max_power" and df[column].dtype == "object":
                power_rpm = df[column].str.extract(r"(\d+\.*\d*)\s*bhp@(\d+)\s*rpm")
                df["max_power_bhp"] = power_rpm[0].astype(float)
                df["max_power_rpm"] = power_rpm[1].astype(int)
                df.drop(column, axis=1, inplace=True)
            elif column == "model" and df[column].dtype == "object":
                df[column] = df[column].str.extract(r"M(\d+)").astype(int)
            elif column == "rear_brakes_type":
                brake_type_mapping = {"Drum": 1, "Disc": 0}
                df[column] = (
                    df[column].map(brake_type_mapping).astype(float).astype(int)
                )
            elif column == "age_of_car":
                df[column] = df[column].apply(convert_age_to_float)
            elif column == "area_cluster":
                df[column] = df[column].apply(extract_number)
            elif column == "fuel_type":
                df[column] = df[column].apply(convert_fuel_type_to_number)
            elif column == "transmission_type":
                df[column] = df[column].apply(convert_transmission_type_to_number)
            elif column == "steering_type":
                df[column] = df[column].apply(convert_steering_type_to_number)
            elif column == "segment":
                df[column] = df[column].apply(convert_segment_to_number)

    dataCleanupWrangling(df_train)
    dataCleanupWrangling(df_test)

    # Adding additional features to the database (both train and test)

    # interaction_term
    df_train["interaction_term"] = df_train["policy_tenure"] * df_train["age_of_car"]
    df_test["interaction_term"] = df_test["policy_tenure"] * df_test["age_of_car"]

    # Car volumne
    df_test["car_volume"] = (
        (df_test["length"] / 1000)
        * (df_test["width"] / 1000)
        * (df_test["height"] / 1000)
    )
    df_train["car_volume"] = (
        (df_train["length"] / 1000)
        * (df_train["width"] / 1000)
        * (df_train["height"] / 1000)
    )

    # Engine Power to Weight Ratio
    df_train["power_to_weight_ratio"] = (
        df_train["max_power_bhp"] / df_train["gross_weight"]
    )
    df_test["power_to_weight_ratio"] = (
        df_test["max_power_bhp"] / df_test["gross_weight"]
    )

    # Fuel Efficiency
    df_train["fuel_efficiency"] = df_train["max_torque_nm"] / df_train["displacement"]
    df_test["fuel_efficiency"] = df_test["max_torque_nm"] / df_test["displacement"]

    # Safety Score
    safety_features = [
        "airbags",
        "is_esc",
        "is_brake_assist",
        "is_rear_window_wiper",
        "is_power_door_locks",
    ]
    df_train["safety_score"] = df_train[safety_features].sum(axis=1)
    df_test["safety_score"] = df_test[safety_features].sum(axis=1)

    # Engine Type Conversion
    df_train["is_turbocharged"] = (
        df_train["engine_type"].apply(lambda x: "Turbo" in x).astype(int)
    )
    df_test["is_turbocharged"] = (
        df_test["engine_type"].apply(lambda x: "Turbo" in x).astype(int)
    )

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # FOLLOWING SECTION FOR INFORMATION ONLY

    # NOTE DATA VISUALISATION AND FEATURE SELECTION PART I

    #  =========================================================================================

    # # Remove 'engine_type' column from the DataFrame
    # delete_col = ['engine_type']
    # df_train.drop(columns=delete_col, inplace=True)

    # # Calculate correlation coefficients between features and the target variable
    # correlations = df_train.corr()['age_of_policyholder'].sort_values(ascending=False)

    # # Initialize lists to store attributes sorted by correlation coefficient and mean squared error (MSE) values
    # attributes_sorted_by_corr = []
    # mse_values = []

    # # Iterate through each column in the DataFrame
    # for label in df_train.columns:
    #     # Exclude the target variable
    #     if label != 'age_of_policyholder':
    #         # Print correlation coefficient between the current feature and the target variable
    #         print(f'Correlation coefficient between {label} and age_of_policyholder: {correlations[label]}')

    #         # Prepare data for linear regression
    #         X = df_train[[label]]
    #         y = df_train['age_of_policyholder']

    #         # Fit linear regression model
    #         reg = LinearRegression()
    #         reg.fit(X, y)
    #         y_pred = reg.predict(X)

    #         # Plot scatter plot and the fitted regression line
    #         plt.scatter(df_train[label], df_train['age_of_policyholder'], color='blue')
    #         plt.plot(df_train[label], y_pred, color='red', linewidth=2)
    #         plt.title(f'Linear Regression: {label} vs Age of Policyholder')
    #         plt.xlabel(label)
    #         plt.ylabel('Age of Policyholder')
    #         plt.show()

    #         # Calculate mean squared error (MSE) and print it
    #         mse = mean_squared_error(y, y_pred)
    #         print(f'Mean Squared Error: {mse}')

    #         # Store attribute name, correlation coefficient, and MSE
    #         attributes_sorted_by_corr.append((label, np.abs(correlations[label]), mse))
    #         mse_values.append(mse)

    # # Sort attributes by maximum absolute correlation coefficient
    # attributes_sorted_by_corr.sort(key=lambda x: x[1], reverse=True)
    # print("Attributes sorted by maximum absolute correlation coefficient:")
    # for attribute, correlation, mse in attributes_sorted_by_corr:
    #     print(f"Attribute: {attribute}, Correlation: {correlation}, MSE: {mse}")

    # # Sort attributes by lowest MSE
    # mse_sorted_attributes = sorted(zip(df_train.columns[:-1], mse_values), key=lambda x: x[1])
    # print("\nAttributes sorted by lowest mean squared error (MSE):")
    # for attribute, mse in mse_sorted_attributes:
    #     print(f"Attribute: {attribute}, MSE: {mse}")

    # # Print attributes sorted by maximum absolute correlation coefficient as a list
    # print("\nAttributes sorted by maximum absolute correlation coefficient (List):")
    # attributes_list = [attribute for attribute, _, _ in attributes_sorted_by_corr]
    # print(attributes_list)

    # Feature selection post reviewing the scatter plots
    # 6 test cases have been carefully selected to assess the MSE
    # Test case 6 has proven to have the lowest MSE

    # test 1
    # features = ['policy_tenure', 'age_of_car','interaction_term', "power_to_weight_ratio"]
    # prediction = ["age_of_policyholder"]

    # test 2
    # features = ['policy_tenure', 'age_of_car', 'area_cluster',
    #        'population_density', 'make', 'segment', 'model', 'fuel_type', 'airbags', 'is_esc', 'is_adjustable_steering', 'is_tpms',
    #        'is_parking_sensors', 'is_parking_camera', 'rear_brakes_type',
    #        'displacement', 'cylinder', 'transmission_type', 'gear_box', 'turning_radius', 'length', 'width', 'height',
    #        'gross_weight', 'is_front_fog_lights', 'is_rear_window_wiper',
    #        'is_rear_window_washer', 'is_rear_window_defogger', 'is_brake_assist',
    #        'is_power_door_locks', 'is_central_locking', 'is_power_steering',
    #        'is_driver_seat_height_adjustable', 'is_day_night_rear_view_mirror',
    #        'is_ecw', 'is_speed_alert', 'ncap_rating', 'is_claim', 'max_torque_nm',
    #        'max_torque_rpm', 'max_power_bhp', 'max_power_rpm', 'car_volume',
    #        'interaction_term', 'power_to_weight_ratio', 'fuel_efficiency',
    #        'safety_score', 'is_turbocharged']
    # prediction = ["age_of_policyholder"]

    # test 3
    # features = ['policy_tenure', 'interaction_term', 'model', 'height', 'segment', 'is_parking_camera', 'max_torque_rpm', 'max_power_rpm', 'age_of_car',
    #            'is_power_door_locks', 'is_central_locking', 'is_ecw', 'make', 'is_rear_window_defogger']
    # prediction = ["age_of_policyholder"]

    # test 4
    # features = ['policy_tenure', 'interaction_term', 'model', 'height', 'segment', 'is_parking_camera']
    # prediction = ["age_of_policyholder"]

    # test 5
    # 91.03
    # features = ['policy_tenure', 'interaction_term', 'model', 'height', 'segment', 'is_parking_camera',
    #             'max_torque_rpm', 'max_power_rpm', 'age_of_car', 'is_power_door_locks', 'is_central_locking', 'is_ecw', 'make',
    #             'is_rear_window_defogger', 'ncap_rating', 'is_brake_assist', 'is_claim', 'area_cluster', 'is_power_steering',
    #             'transmission_type', 'is_parking_sensors', 'car_volume', 'displacement', 'length', 'is_esc', 'fuel_type',
    #             'is_adjustable_steering', 'power_to_weight_ratio', 'is_front_fog_lights', 'turning_radius', 'max_torque_nm']
    # prediction = ["age_of_policyholder"]

    #  =========================================================================================

    # NOTE DATA VISUALISATION AND FEATURE SELECTION PART II

    #  =========================================================================================
    # # hist plots to check the relevancy of the attributes to lodging claims

    # # Adjust plot size
    # plt.figure(figsize=(12, 3))

    # for label in df_train.columns:
    #     # Create histogram plot for claims
    #     plt.hist(df_train[df_train["is_claim"]==1][label], color="blue", label="Claim", alpha=0.5, density=True, edgecolor='black', bins=20)

    #     # Add labels and title
    #     plt.title(f'Histogram of {label} for Claims')
    #     plt.xlabel(label)
    #     plt.ylabel("Probability")

    #     # Show legend
    #     plt.legend()

    #     # Add grid
    #     plt.grid(True)

    #     # Show plot
    #     plt.show()

    # # Countplots to check the relevancy of the attributes to lodging claims

    # # Adjust plot size
    # plt.figure(figsize=(12, 3))

    # for label in df_train.columns:
    #     # Create countplot
    #     sns.countplot(x="is_claim", hue=label, data=df_train, palette='Set2')

    #     # Add labels and title
    #     plt.title(f'Countplot of Claims by {label}')
    #     plt.xlabel('Claim Status')
    #     plt.ylabel('Count')

    #     plt.xticks()

    #     # Show legend outside the plot
    #     plt.legend(title=label, bbox_to_anchor=(1, 1))

    #     # Show plot
    #     plt.show()

    # Feature selection post reviewing the histogram and countplots
    # 6 test cases have been carefully selected to assess the Macro Average F1-Score
    # test case 6 has proven to have the highest Macro Average F1-Score

    # # test 1
    # features = ['is_esc', 'is_adjustable_steering', 'is_tpms', 'is_parking_sensors', 'is_parking_camera',
    #             'is_front_fog_lights', 'is_rear_window_wiper', 'is_rear_window_washer', 'is_rear_window_defogger',
    #             'is_brake_assist', 'is_power_door_locks', 'is_central_locking', 'is_power_steering', 'is_driver_seat_height_adjustable',
    #             'is_day_night_rear_view_mirror', 'is_ecw', 'is_speed_alert' ]
    # prediction = ["is_claim"]

    # # test 2
    # features = ['policy_tenure', 'age_of_car', 'age_of_policyholder']
    # prediction = ["is_claim"]

    # # test 3
    # features = ['policy_tenure', 'age_of_car', 'age_of_policyholder', 'population_density', 'make', 'model',
    #             'airbags', 'is_esc', 'is_adjustable_steering', 'is_tpms', 'is_parking_sensors', 'is_parking_camera',
    #             'displacement', 'transmission_type', 'gear_box', 'steering_type', 'turning_radius', 'is_front_fog_lights', 'is_rear_window_wiper',
    #             'is_rear_window_washer', 'is_rear_window_defogger', 'is_brake_assist', 'is_power_door_locks', 'is_central_locking',
    #             'is_power_steering', 'is_driver_seat_height_adjustable', 'is_day_night_rear_view_mirror', 'is_ecw', 'is_speed_alert', 'ncap_rating'
    #             , 'max_torque_nm', 'max_power_bhp', 'car_volume', 'interaction_term', 'power_to_weight_ratio',
    #             'fuel_efficiency', 'safety_score']
    # prediction = ["is_claim"]

    # # test 4
    # features = ['policy_tenure', 'age_of_car', 'age_of_policyholder', 'area_cluster', 'population_density', 'make', 'segment', 'model', 'fuel_type',
    #             'airbags', 'is_esc', 'is_adjustable_steering', 'is_tpms', 'is_parking_sensors', 'is_parking_camera',
    #             'rear_brakes_type', 'displacement', 'cylinder', 'transmission_type', 'gear_box', 'steering_type', 'turning_radius', 'length',
    #             'width', 'height', 'gross_weight', 'is_front_fog_lights', 'is_rear_window_wiper', 'is_rear_window_washer', 'is_rear_window_defogger',
    #             'is_brake_assist', 'is_power_door_locks', 'is_central_locking', 'is_power_steering', 'is_driver_seat_height_adjustable',
    #             'is_day_night_rear_view_mirror', 'is_ecw', 'is_speed_alert', 'ncap_rating'
    #             , 'max_torque_nm', 'max_torque_rpm', 'max_power_bhp', 'max_power_rpm', 'car_volume',
    #             'interaction_term', 'power_to_weight_ratio',
    #             'fuel_efficiency', 'safety_score']
    # prediction = ["is_claim"]

    # # test 5
    # features = ['policy_tenure', 'age_of_car', 'age_of_policyholder', 'area_cluster', 'population_density',
    #  'make', 'segment', 'model', 'fuel_type', 'airbags', 'is_esc', 'is_adjustable_steering',
    #  'is_tpms', 'is_parking_sensors', 'is_parking_camera', 'rear_brakes_type', 'displacement', 'cylinder',
    #  'transmission_type', 'gear_box', 'steering_type', 'turning_radius', 'length', 'width', 'height',
    #  'gross_weight', 'is_front_fog_lights', 'is_rear_window_wiper', 'is_rear_window_washer', 'is_rear_window_defogger',
    #  'is_brake_assist', 'is_power_door_locks', 'is_central_locking', 'is_power_steering', 'is_driver_seat_height_adjustable',
    #  'is_day_night_rear_view_mirror', 'is_ecw', 'is_speed_alert', 'ncap_rating', 'max_torque_nm', 'max_torque_rpm',
    #  'max_power_bhp', 'max_power_rpm', 'car_volume', 'interaction_term', 'power_to_weight_ratio', 'fuel_efficiency', 'safety_score',
    #  'is_turbocharged']
    # prediction = ["is_claim"]

    #  =========================================================================================
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # def handle_outliers(df, column):
    #     # Calculate IQR
    #     Q1 = df[column].quantile(0.25)
    #     Q3 = df[column].quantile(0.75)
    #     IQR = Q3 - Q1

    #     # Define upper and lower bounds
    #     lower_bound = Q1 - 1.5 * IQR
    #     upper_bound = Q3 + 1.5 * IQR

    #     # Replace outliers with median
    #     df[column] = np.where(df[column] < lower_bound, df[column].median(), df[column])
    #     df[column] = np.where(df[column] > upper_bound, df[column].median(), df[column])

    # feature selection
    features_q1 = [
        "policy_tenure",
        "interaction_term",
        "age_of_car",
        "area_cluster",
        "car_volume",
    ]
    prediction_q1 = ["age_of_policyholder"]

    X_train_q1 = df_train[features_q1]
    X_test_q1 = df_test[features_q1]
    y_train_q1 = df_train[prediction_q1]
    y_test_q1 = df_test[prediction_q1]

    # for feature in features_q1:
    #     handle_outliers(df_train, feature)
    #     handle_outliers(df_test, feature)

    # test 6
    features_q2 = [
        "policy_tenure",
        "age_of_car",
        "age_of_policyholder",
        "area_cluster",
        "population_density",
        "segment",
        "model",
        "fuel_type",
        "airbags",
        "is_esc",
        "is_adjustable_steering",
        "is_tpms",
        "is_parking_sensors",
        "is_parking_camera",
        "rear_brakes_type",
        "displacement",
        "cylinder",
        "transmission_type",
        "gear_box",
        "steering_type",
        "turning_radius",
        "length",
        "width",
        "height",
        "gross_weight",
        "is_front_fog_lights",
        "is_rear_window_wiper",
        "is_rear_window_washer",
        "is_rear_window_defogger",
        "is_brake_assist",
        "is_power_door_locks",
        "is_central_locking",
        "is_power_steering",
        "is_driver_seat_height_adjustable",
        "is_day_night_rear_view_mirror",
        "is_ecw",
        "is_speed_alert",
        "ncap_rating",
        "max_torque_nm",
        "max_torque_rpm",
        "max_power_bhp",
        "max_power_rpm",
        "car_volume",
        "interaction_term",
        "power_to_weight_ratio",
        "fuel_efficiency",
        "safety_score",
    ]
    prediction_q2 = ["is_claim"]

    # for feature in features_q2:
    #     handle_outliers(df_train, feature)
    #     handle_outliers(df_test, feature)

    X_train_q2 = df_train[features_q2]
    X_test_q2 = df_test[features_q2]
    y_train_q2 = df_train[prediction_q2]
    y_test_q2 = df_test[prediction_q2]

    # NOTE PART I REGRESSION

    #  =========================================================================================

    # Train GradientBoostingRegressor model
    boosting_reg_q1 = GradientBoostingRegressor()
    boosting_reg_q1.fit(X_train_q1, y_train_q1)

    # Make predictions and evaluate model
    y_pred_boosting_q1 = boosting_reg_q1.predict(X_test_q1)
    mse_boosting_q1 = int(mean_squared_error(y_test_q1, y_pred_boosting_q1))
    print("Mean Squared Error (Gradient Boosting Regressor):", mse_boosting_q1)

    # Create a DataFrame to store policy_id and predicted age
    output_df_q1 = pd.DataFrame()
    output_df_q1["policy_id"] = df_test_policyid

    # Add predicted age using GradientBoostingRegressor
    output_df_q1["age"] = y_pred_boosting_q1

    # Save the DataFrame to a CSV file
    output_df_q1.to_csv("z5437291.PART1.output.csv", index=False)

    print("Output file z5437291.PART1.output.csv generated successfully.")

    # NOTE PART II CLASSIFICATION

    #  =========================================================================================

    # Feature scaling
    scaler_q2 = StandardScaler()
    X_train_scaled_q2 = scaler_q2.fit_transform(X_train_q2)
    X_test_scaled_q2 = scaler_q2.transform(X_test_q2)  # Separate test data scaling

    # Resample the data (oversampling the minority class)
    ros_q2 = RandomOverSampler(random_state=42)  # Set random_state for reproducibility
    X_train_scaled_resampled_q2, y_train_resampled_q2 = ros_q2.fit_resample(
        X_train_scaled_q2, y_train_q2
    )

    # Create base models with fixed random state
    logistic_model_q2 = LogisticRegression(random_state=42)
    rf_model_q2 = RandomForestClassifier(n_estimators=3, random_state=42)

    # Create a voting classifier
    voting_model_q2 = VotingClassifier(
        estimators=[("lr", logistic_model_q2), ("rf", rf_model_q2)], voting="soft"
    )

    # Evaluate using cross-validation
    cv_scores_q2 = cross_val_score(
        voting_model_q2,
        X_train_scaled_resampled_q2,
        y_train_resampled_q2,
        cv=5,
        scoring="f1_macro",
    )

    # Fit the model on the entire training data
    voting_model_q2.fit(X_train_scaled_resampled_q2, y_train_resampled_q2)

    # Predictions on test data
    y_pred_test_q2 = voting_model_q2.predict(X_test_scaled_q2)

    # # Evaluate on test data
    # print(classification_report(y_test_q2, y_pred_test_q2))

    # Calculate F1 score
    f1_q2 = f1_score(y_test_q2, y_pred_test_q2, average="macro")

    print("F1 Score:", f1_q2)
    # print("Cross-Validation Scores:", cv_scores_q2)

    # Create a DataFrame to store policy_id and predicted age
    output_df_q2 = pd.DataFrame()
    output_df_q2["policy_id"] = df_test_policyid

    # Add predicted age using GradientBoostingRegressor
    output_df_q2["is_claim"] = y_pred_test_q2

    # Save the DataFrame to a CSV file
    output_df_q2.to_csv("z5437291.PART2.output.csv", index=False)

    print("Output file z5437291.PART2.output.csv generated successfully.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 {} <train_file> <test_file>".format(sys.argv[0]))
        sys.exit(1)
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    main(train_file, test_file)
