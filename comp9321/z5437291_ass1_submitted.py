#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# Third-party libraries
# NOTE: You may **only** use the following third-party libraries:
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from thefuzz import fuzz
from thefuzz import process


# NOTE: It isn't necessary to use all of these to complete the assignment,
# but you are free to do so, should you choose.

# Standard libraries
# NOTE: You may use **any** of the Python 3.11 or Python 3.12 standard libraries:
# https://docs.python.org/3.11/library/index.html
# https://docs.python.org/3.12/library/index.html
from pathlib import Path

# ... import your standard libraries here ...
import os
import re

######################################################
# NOTE: DO NOT MODIFY THE LINE BELOW ...
######################################################
studentid = Path(__file__).stem


######################################################
# NOTE: DO NOT MODIFY THE FUNCTION BELOW ...
######################################################
def log(question, output_df, other):
    print(f"--------------- {question}----------------")

    if other is not None:
        print(question, other)
    if output_df is not None:
        df = output_df.head(5).copy(True)
        for c in df.columns:
            df[c] = df[c].apply(lambda a: a[:20] if isinstance(a, str) else a)

        df.columns = [a[:10] + "..." for a in df.columns]
        print(df.to_string())


######################################################
# NOTE: YOU MAY ADD ANY HELPER FUNCTIONS BELOW ...
######################################################

# additional functions are used for question 10 but have been included in in question_10() function for readability

######################################################
# QUESTIONS TO COMPLETE BELOW ...
######################################################


######################################################
# NOTE: DO NOT MODIFY THE FUNCTION SIGNATURE BELOW ...
######################################################
def question_1(jobs_csv):
    """Read the data science jobs CSV file into a DataFrame.

    See the assignment spec for more details.

    Args:
        jobs_csv (str): Path to the jobs CSV file.

    Returns:
        DataFrame: The jobs DataFrame.
    """

    ######################################################
    # TODO: Your code goes here ...
    ######################################################

    # Read the CSV file into a DataFrame
    df = pd.read_csv(jobs_csv)

    ######################################################
    # NOTE: DO NOT MODIFY THE CODE BELOW ...
    ######################################################
    log("QUESTION 1", output_df=df, other=df.shape)
    return df


######################################################
# NOTE: DO NOT MODIFY THE FUNCTION SIGNATURE BELOW ...
######################################################
def question_2(cost_csv, cost_url):
    """Read the cost of living CSV into a DataFrame.  If the CSV file does not
    exist, scrape it from the specified URL and save it to the CSV file.

    See the assignment spec for more details.

    Args:
        cost_csv (str): Path to the cost of living CSV file.
        cost_url (str): URL of the cost of living page.

    Returns:
        DataFrame: The cost of living DataFrame.
    """

    ######################################################
    # TODO: Your code goes here ...
    ######################################################
    # Check if the CSV file exists
    if os.path.exists(cost_csv):
        # If the CSV file exists, load the data from the CSV file
        df = pd.read_csv(cost_csv)
    else:
        # If the CSV file doesn't exist, scrape the website
        tables = pd.read_html(cost_url)
        df = tables[0]  # Assuming the first table on the page contains the desired data

        # Sanitise column names
        df.columns = df.columns.str.lower().str.replace(" ", "_")

        df.to_csv(cost_csv, index=False)

    ######################################################
    # NOTE: DO NOT MODIFY THE CODE BELOW ...
    ######################################################
    log("QUESTION 2", output_df=df, other=df.shape)
    return df


######################################################
# NOTE: DO NOT MODIFY THE FUNCTION SIGNATURE BELOW ...
######################################################
def question_3(currency_csv, currency_url):
    """Read the currency conversion rates CSV into a DataFrame. If the CSV
    file does not exist, scrape it from the specified URL and save it to
    the CSV file.

    Args:
        currency_csv (str): Path to the currency conversion rates CSV file.
        currency_url (str): URL of the currency conversion rates page.

    Returns:
        DataFrame: The currency conversion rates DataFrame.
    """

    ######################################################
    # TODO: Your code goes here ...
    ######################################################
    # Check if the CSV file exists
    if os.path.exists(currency_csv):
        # If the CSV file exists, load the data from the CSV file
        df = pd.read_csv(currency_csv)
    else:
        # If the CSV file doesn't exist, scrape the website
        tables = pd.read_html(currency_url, header=0)
        df = tables[0]  # Assuming the first table on the page contains the desired data

        # Replace all non-breaking spaces with regular spaces in column names
        df.columns = [col.replace("\xa0", " ") for col in df.columns]

        # Replace all non-breaking spaces with regular spaces in data
        for col in df.columns:
            if df[col].dtype == "object":  # Check if the column contains strings
                df[col] = df[col].str.replace("\xa0", " ")

        # Remove all columns under "Nearest actual exchange rate" along with the headers
        df = df.iloc[:, [0, 2, 5]]

        # Assign the first row as column names
        df.columns = df.iloc[0]

        df = df.drop(0)

        # Convert column names to lowercase
        df.columns = map(str.lower, df.columns)

        df = df.rename(columns={("31 Dec 23").lower(): "rates"})

        df.to_csv(currency_csv, index=False)

    ######################################################
    # NOTE: DO NOT MODIFY THE CODE BELOW ...
    ######################################################
    log("QUESTION 3", output_df=df, other=df.shape)
    return df


######################################################
# NOTE: DO NOT MODIFY THE FUNCTION SIGNATURE BELOW ...
######################################################
def question_4(country_csv, country_url):
    """Read the country codes CSV into a DataFrame.  If the CSV file does not
    exist, it will be scrape the data from the specified URL and save it to the
    CSV file.

    See the assignment spec for more details.

    Args:
        cost_csv (str): Path to the country codes CSV file.
        cost_url (str): URL of the country codes page.

    Returns:
        DataFrame: The country codes DataFrame.
    """

    ######################################################
    # TODO: Your code goes here ...
    ######################################################
    # Check if the CSV file exists
    if os.path.exists(country_csv):
        # If the CSV file exists, read it into a DataFrame
        df = pd.read_csv(country_csv)
    else:
        # If the CSV file does not exist, scrape the data from the website
        tables = pd.read_html(country_url, header=0)
        df = tables[0]

        # Remove irrelevant columns
        df.drop(columns=["Year", "ccTLD", "Notes"], inplace=True)

        # Rename columns
        df.rename(columns={"Country name": "country", "Code": "code"}, inplace=True)

        df.to_csv(country_csv, index=False)
    ######################################################
    # NOTE: DO NOT MODIFY THE CODE BELOW ...
    ######################################################
    log("QUESTION 4", output_df=df, other=df.shape)
    return df


######################################################
# NOTE: DO NOT MODIFY THE FUNCTION SIGNATURE BELOW ...
######################################################
def question_5(jobs_df):
    """Summarise some dimensions of the jobs DataFrame.

    See the assignment spec for more details.

    Args:
        jobs_df (DataFrame): The jobs DataFrame returned in question 1.

    Returns:
        DataFrame: The summary DataFrame.
    """

    ######################################################
    # TODO: Your code goes here ...
    ######################################################
    df = pd.DataFrame(
        index=jobs_df.columns, columns=["observations", "distinct", "missing"]
    )

    for column in jobs_df.columns:
        observations = jobs_df[column].count()  # Count non-missing values
        distinct = jobs_df[column].nunique()  # Count distinct non-missing values
        missing = jobs_df[column].isnull().sum()  # Count missing values

        # Populate the new DataFrame with the calculated values
        df.loc[column] = [observations, distinct, missing]

    ######################################################
    # NOTE: DO NOT MODIFY THE CODE BELOW ...
    ######################################################
    log("QUESTION 5", output_df=df, other=df.shape)
    return df


######################################################
# NOTE: DO NOT MODIFY THE FUNCTION SIGNATURE BELOW ...
######################################################
def question_6(jobs_df):
    """Add an experience rating column to the jobs DataFrame.

    See the assignment spec for more details.

    Args:
        jobs_df (DataFrame): The jobs DataFrame returned in question 1.

    Returns:
        DataFrame: The jobs DataFrame with the experience rating column added.
    """

    ######################################################
    # TODO: Your code goes here ...
    ######################################################
    # Define a mapping of experience levels to ratings
    experience_rating_map = {"EN": 1, "MI": 2, "SE": 3, "EX": 4}

    jobs_df["experience_rating"] = jobs_df["experience_level"].map(
        experience_rating_map
    )

    df = jobs_df

    ######################################################
    # NOTE: DO NOT MODIFY THE CODE BELOW ...
    ######################################################
    log("QUESTION 6", output_df=df, other=df.shape)
    return df


######################################################
# NOTE: DO NOT MODIFY THE FUNCTION SIGNATURE BELOW ...
######################################################
def question_7(jobs_df, country_df):
    """Merge the jobs and country codes DataFrames.

    See the assignment spec for more details.

    Args:
        jobs_df (DataFrame): The jobs DataFrame returned in question 6.
        country_df (DataFrame): The country codes DataFrame returned in
                                question 4.

    Returns:
        DataFrame: The merged DataFrame.
    """

    ######################################################
    # TODO: Your code goes here ...
    ######################################################
    # Merge the jobs DataFrame with the country codes DataFrame based on the employee_residence column
    df = jobs_df.merge(
        country_df, left_on="employee_residence", right_on="code", how="left"
    )

    df.drop(columns=["code"], inplace=True)
    df = df.rename(columns={"Country name (using title case)": "country"})

    ######################################################
    # NOTE: DO NOT MODIFY THE CODE BELOW ...
    ######################################################
    log("QUESTION 7", output_df=df, other=df.shape)

    return df


######################################################
# NOTE: DO NOT MODIFY THE FUNCTION SIGNATURE BELOW ...
######################################################
def question_8(jobs_df, currency_df):
    """Add an Australian dollar salary column to the jobs DataFrame.

    See the assignment spec for more details.

    Args:
        jobs_df (DataFrame): The jobs DataFrame returned in question 7.
        currency_df (DataFrame): The currency conversion rates DataFrame
                                 returned in question 3.

    Returns:
        DataFrame: The jobs DataFrame with the Australian dollar salary column
                   added.
    """

    ######################################################
    # TODO: Your code goes here ...
    ######################################################
    # Read the currency exchange rates DataFrame
    df_exchange_rate = currency_df

    df = jobs_df

    # Extract the conversion rate for USD to AUD
    us_rate_row = df_exchange_rate[df_exchange_rate["country"] == "United States"]
    conversion_rate = us_rate_row["rates"].values[0]
    conversion_rate_float = float(conversion_rate)

    # Calculate the salary in Australian dollars
    df["exchange_rate"] = conversion_rate_float
    df["salary_in_aus"] = (df["salary_in_usd"] / df["exchange_rate"]).astype(int)

    df = df[df["work_year"] == 2023].copy()  # Create a copy of the DataFrame

    # Drop the temporary exchange_rate column
    df.drop(columns=["exchange_rate"], inplace=True)

    ######################################################
    # NOTE: DO NOT MODIFY THE CODE BELOW ...
    ######################################################
    log("QUESTION 8", output_df=df, other=df.shape)
    return df


######################################################
# NOTE: DO NOT MODIFY THE FUNCTION SIGNATURE BELOW ...
######################################################
def question_9(cost_df):
    """Re-scale the cost of living DataFrame to be relative to Australia.

    See the assignment spec for more details.

    Args:
        cost_df (DataFrame): The cost of living DataFrame returned in question 2.

    Returns:
        DataFrame: The re-scaled cost of living DataFrame.
    """

    ######################################################
    # TODO: Your code goes here ...
    ######################################################

    df = cost_df.copy()

    # Keep only 'country' and 'cost_of_living_plus_rent_index' columns
    df = df[["country", "cost_of_living_plus_rent_index"]]

    # Find the index for Australia
    index_australia = df[df["country"] == "Australia"][
        "cost_of_living_plus_rent_index"
    ].values[0]

    # Calculate the re-scaled index for each country
    df["cost_of_living_plus_rent_index"] = (
        df["cost_of_living_plus_rent_index"] / index_australia
    ) * 100

    # Round the calculated values to 1 decimal place using .loc
    df.loc[:, "cost_of_living_plus_rent_index"] = df[
        "cost_of_living_plus_rent_index"
    ].round(1)

    # Sort the DataFrame by increasing cost_of_living_plus_rent_index
    df = df.sort_values(by="cost_of_living_plus_rent_index")

    ######################################################
    # NOTE: DO NOT MODIFY THE CODE BELOW ...
    ######################################################
    log("QUESTION 9", output_df=df, other=df.shape)
    return df


######################################################
# NOTE: DO NOT MODIFY THE FUNCTION SIGNATURE BELOW ...
######################################################
def question_10(jobs_df, cost_df):
    """Merge the jobs and cost of living DataFrames.

    See the assignment spec for more details.

    Args:
        jobs_df (DataFrame): The jobs DataFrame returned in question 8.
        cost_df (DataFrame): The cost of living DataFrame returned in question 9.

    Returns:
        DataFrame: The merged DataFrame.
    """

    ######################################################
    # TODO: Your code goes here ...
    ######################################################

    # Load the CSV files into DataFrames
    df_question_8 = jobs_df
    df_question_9 = cost_df

    # Preprocess country names
    def preprocess_country_name(country):
        # Convert to lowercase and remove special characters
        country = country.lower()
        country = re.sub(r"[^a-z\s]", "", country)
        return country

    df_question_8["country_processed"] = df_question_8["country"].apply(
        preprocess_country_name
    )
    df_question_9["country_processed"] = df_question_9["country"].apply(
        preprocess_country_name
    )

    # Define a function to perform fuzzy matching
    def fuzzy_match_country(country1, country_list):
        best_match = process.extractOne(country1, country_list, score_cutoff=90)
        if best_match is not None:
            return best_match[0]
        return None

    # Perform fuzzy matching and merge the DataFrames
    df_question_8["matched_country"] = df_question_8["country_processed"].apply(
        fuzzy_match_country, args=(df_question_9["country_processed"],)
    )
    df = pd.merge(
        df_question_8,
        df_question_9,
        how="left",
        left_on="matched_country",
        right_on="country_processed",
    )

    # Filter out rows with no match
    df = df.dropna(subset=["matched_country"])

    # Rename the 'cost_of_living_plus_rent_index' column to 'cost_of_living'
    df = df.rename(columns={"cost_of_living_plus_rent_index": "cost_of_living"})

    df.drop(
        ["country_processed_x", "matched_country", "country_y", "country_processed_y"],
        axis=1,
        inplace=True,
    )
    df.rename(columns={"country_x": "country"}, inplace=True)

    ######################################################
    # NOTE: DO NOT MODIFY THE CODE BELOW ...
    ######################################################
    log("QUESTION 10", output_df=df, other=df.shape)
    return df


######################################################
# NOTE: DO NOT MODIFY THE FUNCTION SIGNATURE BELOW ...
######################################################
def question_11(jobs_df):
    """Create a pivot table of the average salary in AUD by country and
    experience rating.

    See the assignment spec for more details.

    Args:
        jobs_df (DataFrame): The jobs DataFrame returned in question 10.

    Returns:
        DataFrame: The pivot table.
    """

    df = pd.pivot_table(
        jobs_df,
        values="salary_in_aus",
        index="country",
        columns="experience_rating",
        aggfunc="mean",
        fill_value=0,
    )

    # Convert all values to integers
    df = df.astype(int)

    column_order = [1, 2, 3, 4]

    # Sort columns based on the desired order
    df = df[column_order]

    # Rename columns to match the specified format
    df.columns = [("salary_in_aud", rating) for rating in column_order]

    ######################################################
    # NOTE: DO NOT MODIFY THE CODE BELOW ...
    ######################################################
    log("QUESTION 11", output_df=None, other=df)
    return df


######################################################
# NOTE: DO NOT MODIFY THE FUNCTION SIGNATURE BELOW ...
######################################################
def question_12(jobs_df):
    """Create a visualization of data science jobs to help inform a decision
    about where to live, based (minimally) on salary and cost of living.

    Args:
        jobs_df (DataFrame): The jobs DataFrame returned in question 10.
    """

    # Group by country and calculate mean salary and cost of living
    country_stats = (
        jobs_df.groupby("country")
        .agg({"salary_in_aus": "mean", "cost_of_living": "mean"})
        .reset_index()
    )

    # Remove outliers
    country_stats = country_stats[
        country_stats["salary_in_aus"] < country_stats["salary_in_aus"].quantile(0.95)
    ]

    # Sort by salary_in_aus
    country_stats = country_stats.sort_values(by="salary_in_aus", ascending=False)

    # Select top 5 countries including Australia
    top_countries = country_stats.head(5)

    # If Australia is not in the top 5, include it
    if "Australia" not in top_countries["country"].values:
        australia_stats = country_stats[country_stats["country"] == "Australia"]
        top_countries = pd.concat([top_countries, australia_stats])

    # Create a figure and subplots
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(18, 18))

    # Scatter plot of Average Salary vs. Cost of Living
    scatter = axes[0, 0].scatter(
        top_countries["salary_in_aus"],
        top_countries["cost_of_living"],
        s=top_countries["salary_in_aus"] / 100,  # Adjusting marker size
        color=["orange", "brown", "yellow", "orchid", "lightcoral", "green"],
        alpha=0.8,  # Adding transparency
        edgecolors="black",  # Improving marker edges
        linewidths=2,
    )
    axes[0, 0].set_title("Average Salary vs. Cost of Living", fontsize=18)
    axes[0, 0].set_xlabel("Average Salary (AUD)", fontsize=16)
    axes[0, 0].set_ylabel("Average Cost of Living", fontsize=16)
    axes[0, 0].tick_params(axis="both", which="major", labelsize=12)
    axes[0, 0].grid(True)

    # Extend the plot boundaries
    x_extension = 0.15  # 15% extension beyond the data range for x-axis
    y_extension = 0.15  # 15% extension beyond the data range for y-axis

    # Calculate the extension values for the x-axis and y-axis
    x_range = (
        top_countries["salary_in_aus"].max() - top_countries["salary_in_aus"].min()
    )
    y_range = (
        top_countries["cost_of_living"].max() - top_countries["cost_of_living"].min()
    )

    # Extend the x-axis boundary
    x_min = top_countries["salary_in_aus"].min() - x_extension * x_range
    x_max = top_countries["salary_in_aus"].max() + x_extension * x_range
    axes[0, 0].set_xlim(x_min, x_max)

    # Extend the y-axis boundary
    y_min = top_countries["cost_of_living"].min() - y_extension * y_range
    y_max = top_countries["cost_of_living"].max() + y_extension * y_range
    axes[0, 0].set_ylim(y_min, y_max)

    # Adding country labels with offset
    label_offset_x = 0  # Adjust this value to move the label along the x-axis
    label_offset_y = 10  # Adjust this value to move the label along the y-axis
    for index, row in top_countries.iterrows():
        axes[0, 0].text(
            row["salary_in_aus"] + label_offset_x,
            row["cost_of_living"] + label_offset_y,
            row["country"],
            fontsize=14,
            ha="center",
        )

    # Bar plot of Average Salary by Country
    axes[0, 1].bar(
        top_countries["country"],
        top_countries["salary_in_aus"],
        color=["orange", "brown", "yellow", "orchid", "lightcoral", "green"],
    )
    axes[0, 1].set_title("Average Salary by Country", fontsize=18)
    axes[0, 1].set_xlabel("Country", fontsize=16)
    axes[0, 1].set_ylabel("Average Salary (AUD)", fontsize=16)
    axes[0, 1].tick_params(axis="x", rotation=25)
    axes[0, 1].tick_params(axis="y", which="major", labelsize=16)

    # Add a horizontal line representing global average salary
    global_avg_salary = jobs_df["salary_in_aus"].mean()
    axes[0, 1].axhline(
        global_avg_salary, color="black", linestyle="dashdot", linewidth=2
    )
    axes[0, 1].text(
        len(top_countries) - 4.5,
        global_avg_salary + 5000,
        f"Global Avg: ${global_avg_salary:.0f}",
        fontsize=14,
        color="black",
    )
    # Add individual salary labels
    for index, value in enumerate(top_countries["salary_in_aus"]):
        axes[0, 1].text(
            index,
            value + 2000,
            f"${value/1000:.0f}k",
            ha="center",
            fontsize=12,
            color="black",
        )

    # Histogram of Salary Distribution with Normal Distribution Curve
    data = jobs_df["salary_in_aus"]
    mean = np.mean(data)
    std_dev = np.std(data)
    x = np.linspace(mean - 3 * std_dev, mean + 3 * std_dev, 100)
    y = (1 / (std_dev * np.sqrt(2 * np.pi))) * np.exp(
        -0.5 * ((x - mean) / std_dev) ** 2
    )

    axes[1, 1].hist(
        data, bins=50, density=True, color="lightblue", alpha=0.6, label="Histogram"
    )
    axes[1, 1].plot(x, y, "k--", label="Normal Distribution")

    # Add title and labels
    axes[1, 1].set_title(
        "Salary Distribution with Normal Distribution Curve", fontsize=16
    )
    axes[1, 1].set_xlabel("Salary (AUD)", fontsize=14)
    axes[1, 1].set_ylabel("Density", fontsize=14)
    axes[1, 1].tick_params(axis="both", which="major", labelsize=12)
    axes[1, 1].legend(
        loc="upper right", title="Legend", fontsize=14, title_fontsize=16, shadow=True
    )

    # Pie Chart of Distribution of Average Salary by Country
    axes[1, 0].pie(
        top_countries["salary_in_aus"],
        labels=top_countries["country"],
        autopct="%1.1f%%",
        startangle=140,
        pctdistance=0.8,
        labeldistance=1.1,
        radius=1,
        wedgeprops=dict(edgecolor="black", linewidth=0.5),
        textprops=dict(color="black", fontsize=16),
    )
    axes[1, 0].set_title("Distribution of Average Salary by Country", fontsize=18)

    plt.tight_layout()

    ######################################################
    # NOTE: DO NOT MODIFY THE CODE BELOW ...
    ######################################################
    plt.savefig(f"{studentid}-Q12.png")


######################################################
# NOTE: DO NOT MODIFY THE MAIN FUNCTION BELOW ...
######################################################
if __name__ == "__main__":
    # data ingestion and cleaning
    df1 = question_1("ds_jobs.csv")
    df2 = question_2(
        "cost_of_living.csv",
        "https://www.cse.unsw.edu.au/~cs9321/24T1/ass1/cost_of_living.html",
    )
    df3 = question_3(
        "exchange_rates.csv",
        "https://www.cse.unsw.edu.au/~cs9321/24T1/ass1/exchange_rates.html",
    )
    df4 = question_4(
        "country_codes.csv",
        "https://www.cse.unsw.edu.au/~cs9321/24T1/ass1/country_codes.html",
    )

    # data exploration
    df5 = question_5(df1.copy(True))

    # data manipulation
    df6 = question_6(df1.copy(True))
    df7 = question_7(df6.copy(True), df4.copy(True))
    df8 = question_8(df7.copy(True), df3.copy(True))
    df9 = question_9(df2.copy(True))
    df10 = question_10(df8.copy(True), df9.copy(True))
    df11 = question_11(df10.copy(True))

    # data visualisation
    question_12(df10.copy(True))
