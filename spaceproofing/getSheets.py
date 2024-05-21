#!/usr/bin/env python

# export data sheets from xlsx to csv
import inspect
import os.path
import pandas as pd
import csv

# this python files whole directory
thisFile = inspect.getfile(inspect.currentframe())

# this scripts folder path
directory = os.path.dirname(thisFile)

# uncomment next line for custom folder location
# directory = r"C:\Users\dharman.gersch\OneDrive - Aurecon Group\Desktop\ProfileCatalogue"

myexcelpath = os.path.join(directory, "WHTP2-AAJV-WHT-TU-TG11-REG-000001.xlsx")

print("Reading: {0}...".format(myexcelpath))
xl = pd.ExcelFile(myexcelpath)
xl.sheet_names  # see all sheet names
for sheet_name in xl.sheet_names:
    sheet_directory = os.path.join(directory, sheet_name + ".csv")

    headerIndex = 0
    if "Schedule" in sheet_name:
        headerIndex = 3
        df = xl.parse(
            sheet_name,
            header=headerIndex,
            quoting=csv.QUOTE_ALL,
            skipinitialspace=True,
            dtype=str,
        )  # read a specific sheet to DataFrame

    elif "Profile Catalogue" in sheet_name:
        headerIndex = 1
        df = xl.parse(
            sheet_name,
            header=headerIndex,
            quoting=csv.QUOTE_ALL,
            skipinitialspace=True,
            dtype=str,
        )  # read a specific sheet to DataFrame
        df["ID"] = df["ID"].fillna(
            value="_"
        )  # fill empty ID cells so they can all be measured
        df.drop(
            df[df["ID"].str.len() < 13].index, inplace=True
        )  # remove rows where ID is too short or missing
    else:
        print("skipping {0}".format(sheet_name))
        continue

    df.columns = [x.replace("\n", " ") for x in df.columns.to_list()]
    df.dropna(
        how="all", axis=1, inplace=True
    )  # remove all empty columns (there are lots sometimes from excel)
    df.replace(r"\n", " ", regex=True, inplace=True)  # remove newlines
    df.replace(" ", "", regex=True, inplace=True)  # remove spaces
    df.replace(",", "", regex=True, inplace=True)  # remove spaces
    df.to_csv(sheet_directory, index=False, quoting=csv.QUOTE_ALL)  # write to csv
    print("wrote: {0}".format(sheet_name))

print("Done")
# sheets = []
# workbook = load_workbook( myexcelpath, read_only= True, data_only= True,)
# all_worksheets = workbook.get_sheet_names()
# for worksheet_name in all_worksheets:
#     sheets.append(worksheet_name)

# print(sheets)
# csvFiles = []

# for worksheet_name in sheets:
#         print("Export " + worksheet_name + " ...")

#         try:
#             worksheet = workbook.get_sheet_by_name(worksheet_name)
#         except KeyError:
#             print("Could not find " + worksheet_name)
#             sys.exit(1)

#         your_csv_file = open(''.join([worksheet_name,'.csv']), 'w', newline='')
#         wr = csv.writer(your_csv_file, quoting=csv.QUOTE_ALL)
#         for row in worksheet.iter_rows():
#             lrow = []
#             for cell in row:
#                 lrow.append(cell.value)

#             wr.writerow(lrow)
#         print(" ... done")
#         csvFiles.append(your_csv_file)
#         your_csv_file.close()

# for your_csv_file in csvFiles:
#     df = pd.DataFrame(pd.read_csv(your_csv_file))
#     df.dropna(how='all', axis=1, inplace=True)
