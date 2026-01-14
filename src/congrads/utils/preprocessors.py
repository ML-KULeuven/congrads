"""Preprocessing functions for various datasets.

This module provides preprocessing pipelines for multiple datasets:
- BiasCorrection: Temperature bias correction dataset
- FamilyIncome: Family income and expenses dataset
- AdultCensusIncome: Adult Census Income dataset

Each preprocessing function applies appropriate transformations including
normalization, feature engineering, constraint filtering, and sampling.
"""

import numpy as np
import pandas as pd


def preprocess_BiasCorrection(df: pd.DataFrame) -> pd.DataFrame:  # noqa: N802
    """Preprocesses the given dataframe for bias correction by performing a series of transformations.

    The function sequentially:

    - Drops rows with missing values.
    - Converts a date string to datetime format and adds year, month,
      and day columns.
    - Normalizes the columns with specific logic for input and output variables.
    - Adds a multi-index indicating which columns are input or output variables.
    - Samples 2500 examples from the dataset without replacement.

    Args:
        df (pd.DataFrame): The input dataframe containing the data
            to be processed.

    Returns:
        pd.DataFrame: The processed dataframe after applying
        the transformations.
    """

    def date_to_datetime(df: pd.DataFrame) -> pd.DataFrame:
        """Transform the string that denotes the date to the datetime format in pandas."""
        # make copy of dataframe
        df_temp = df.copy()
        # add new column at the front where the date string is
        # transformed to the datetime format
        df_temp.insert(0, "DateTransformed", pd.to_datetime(df_temp["Date"]))
        return df_temp

    def add_year(df: pd.DataFrame) -> pd.DataFrame:
        """Extract the year from the datetime cell and add it as a new column to the dataframe at the front."""
        # make copy of dataframe
        df_temp = df.copy()
        # extract year and add new column at the front containing these numbers
        df_temp.insert(0, "Year", df_temp["DateTransformed"].dt.year)
        return df_temp

    def add_month(df: pd.DataFrame) -> pd.DataFrame:
        """Extract the month from the datetime cell and add it as a new column to the dataframe at the front."""
        # make copy of dataframe
        df_temp = df.copy()
        # extract month and add new column at index 1 containing these numbers
        df_temp.insert(1, "Month", df_temp["DateTransformed"].dt.month)
        return df_temp

    def add_day(df: pd.DataFrame) -> pd.DataFrame:
        """Extract the day from the datetime cell and add it as a new column to the dataframe at the front."""
        # make copy of dataframe
        df_temp = df.copy()
        # extract day and add new column at index 2 containing these numbers
        df_temp.insert(2, "Day", df_temp["DateTransformed"].dt.day)
        return df_temp

    def add_input_output_temperature(df: pd.DataFrame) -> pd.DataFrame:
        """Add a multiindex denoting if the column is an input or output variable."""
        # copy the dataframe
        temp_df = df.copy()
        # extract all the column names
        column_names = temp_df.columns.tolist()
        # only the last 2 columns are output variables, all others are input
        # variables. So make list of corresponding lengths of
        # 'Input' and 'Output'
        input_list = ["Input"] * (len(column_names) - 2)
        output_list = ["Output"] * 2
        # concat both lists
        input_output_list = input_list + output_list
        # define multi index for attaching this 'Input' and 'Output' list with
        # the column names already existing
        multiindex_bias = pd.MultiIndex.from_arrays([input_output_list, column_names])
        # transpose such that index can be adjusted to multi index
        new_df = pd.DataFrame(df.transpose().to_numpy(), index=multiindex_bias)
        # transpose back such that columns are the same as before
        # except with different labels
        return new_df.transpose()

    def normalize_columns_bias(df: pd.DataFrame) -> pd.DataFrame:
        """Normalize the columns for the bias correction dataset.

        This is different from normalizing all the columns separately
        because the upper and lower bounds for the output variables
        are assumed to be the same.
        """
        # copy the dataframe
        temp_df = df.copy()
        # normalize each column
        for feature_name in df.columns:
            # the output columns are normalized using the same upper and
            # lower bound for more efficient check of the inequality
            if feature_name == "Next_Tmax" or feature_name == "Next_Tmin":
                max_value = 38.9
                min_value = 11.3
            # the input columns are normalized using their respective
            # upper and lower bounds
            else:
                max_value = df[feature_name].max()
                min_value = df[feature_name].min()
            temp_df[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
        return temp_df

    def sample_2500_examples(df: pd.DataFrame) -> pd.DataFrame:
        """Sample 2500 examples from the dataframe without replacement."""
        temp_df = df.copy()
        sample_df = temp_df.sample(n=2500, replace=False, random_state=3, axis=0)
        return sample_df

    return (
        # drop missing values
        df.dropna(how="any")
        # transform string date to datetime format
        .pipe(date_to_datetime)
        # add year as a single column
        .pipe(add_year)
        # add month as a single column
        .pipe(add_month)
        # add day as a single column
        .pipe(add_day)
        # remove original date string and the datetime format
        .drop(["Date", "DateTransformed"], axis=1, inplace=False)
        # convert all numbers to float32
        .astype("float32")
        # normalize columns
        .pipe(normalize_columns_bias)
        # add multi index indicating which columns are corresponding
        # to input and output variables
        .pipe(add_input_output_temperature)
        # sample 2500 examples out of the dataset
        .pipe(sample_2500_examples)
    )


def preprocess_FamilyIncome(df: pd.DataFrame) -> pd.DataFrame:  # noqa: N802
    """Preprocesses the given Family Income dataframe.

    The function sequentially:

    - Drops rows with missing values.
    - Converts object columns to appropriate data types and
      removes string columns.
    - Removes certain unnecessary columns like
      'Agricultural Household indicator' and related features.
    - Adds labels to columns indicating whether they are
      input or output variables.
    - Normalizes the columns individually.
    - Checks and removes rows that do not satisfy predefined constraints
      (household income > expenses, food expenses > sub-expenses).
    - Samples 2500 examples from the dataset without replacement.

    Args:
        df (pd.DataFrame): The input Family Income dataframe containing
            the data to be processed.

    Returns:
        pd.DataFrame: The processed dataframe after applying the
        transformations and constraints.
    """

    def normalize_columns_income(df: pd.DataFrame) -> pd.DataFrame:
        """Normalize each column of the dataframe independently.

        This function scales each column to have values between 0 and 1
        (or another standard normalization, depending on implementation),
        making it suitable for numerical processing. While designed for
        the Family Income dataset, it can be applied to any dataframe
        with numeric columns.

        Args:
            df (pd.DataFrame): Input dataframe to normalize.

        Returns:
            pd.DataFrame: Dataframe with each column normalized independently.
        """
        # copy the dataframe
        temp_df = df.copy()
        # normalize each column
        for feature_name in df.columns:
            max_value = df[feature_name].max()
            min_value = df[feature_name].min()
            temp_df[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
        return temp_df

    def check_constraints_income(df: pd.DataFrame) -> pd.DataFrame:
        """Filter rows that violate income-related constraints.

        This function is specific to the Family Income dataset. It removes rows
        that do not satisfy the following constraints:
            1. Household income must be greater than all expenses.
            2. Food expense must be greater than the sum of detailed food expenses.

        Args:
            df (pd.DataFrame): Input dataframe containing income and expense data.

        Returns:
            pd.DataFrame: Filtered dataframe containing only rows that satisfy
            all constraints.
        """
        temp_df = df.copy()
        # check that household income is larger than expenses in the output
        input_array = temp_df["Input"].to_numpy()
        income_array = np.add(
            np.multiply(
                input_array[:, [0, 1]],
                np.subtract(np.asarray([11815988, 9234485]), np.asarray([11285, 0])),
            ),
            np.asarray([11285, 0]),
        )
        expense_array = temp_df["Output"].to_numpy()
        expense_array = np.add(
            np.multiply(
                expense_array,
                np.subtract(
                    np.asarray(
                        [
                            791848,
                            437467,
                            140992,
                            74800,
                            2188560,
                            1049275,
                            149940,
                            731000,
                        ]
                    ),
                    np.asarray([3704, 0, 0, 0, 1950, 0, 0, 0]),
                ),
            ),
            np.asarray([3704, 0, 0, 0, 1950, 0, 0, 0]),
        )
        expense_array_without_dup = expense_array[:, [0, 4, 5, 6, 7]]
        sum_expenses = np.sum(expense_array_without_dup, axis=1)
        total_income = np.sum(income_array, axis=1)
        sanity_check_array = np.greater_equal(total_income, sum_expenses)
        temp_df["Unimportant"] = sanity_check_array.tolist()
        reduction = temp_df[temp_df.Unimportant]
        drop_reduction = reduction.drop("Unimportant", axis=1)

        # check that the food expense is larger than all the sub expenses
        expense_reduced_array = drop_reduction["Output"].to_numpy()
        expense_reduced_array = np.add(
            np.multiply(
                expense_reduced_array,
                np.subtract(
                    np.asarray(
                        [
                            791848,
                            437467,
                            140992,
                            74800,
                            2188560,
                            1049275,
                            149940,
                            731000,
                        ]
                    ),
                    np.asarray([3704, 0, 0, 0, 1950, 0, 0, 0]),
                ),
            ),
            np.asarray([3704, 0, 0, 0, 1950, 0, 0, 0]),
        )
        food_mul_expense_array = expense_reduced_array[:, [1, 2, 3]]
        food_mul_expense_array_sum = np.sum(food_mul_expense_array, axis=1)
        food_expense_array = expense_reduced_array[:, 0]
        sanity_check_array = np.greater_equal(food_expense_array, food_mul_expense_array_sum)
        drop_reduction["Unimportant"] = sanity_check_array.tolist()
        new_reduction = drop_reduction[drop_reduction.Unimportant]
        satisfied_constraints_df = new_reduction.drop("Unimportant", axis=1)

        return satisfied_constraints_df

    def add_input_output_family_income(df: pd.DataFrame) -> pd.DataFrame:
        """Add a multiindex denoting if the column is an input or output variable."""
        # copy the dataframe
        temp_df = df.copy()
        # extract all the column names
        column_names = temp_df.columns.tolist()
        # the 2nd-9th columns correspond to output variables and all
        # others to input variables. So make list of corresponding
        # lengths of 'Input' and 'Output'
        input_list_start = ["Input"]
        input_list_end = ["Input"] * (len(column_names) - 9)
        output_list = ["Output"] * 8
        # concat both lists
        input_output_list = input_list_start + output_list + input_list_end
        # define multi index for attaching this 'Input' and
        # 'Output' list with the column names already existing
        multiindex_bias = pd.MultiIndex.from_arrays([input_output_list, column_names])
        # transpose such that index can be adjusted to multi index
        new_df = pd.DataFrame(df.transpose().to_numpy(), index=multiindex_bias)
        # transpose back such that columns are the same as
        # before except with different labels
        return new_df.transpose()

    def sample_2500_examples(df: pd.DataFrame) -> pd.DataFrame:
        """Sample 2500 examples from the dataframe without replacement."""
        temp_df = df.copy()
        sample_df = temp_df.sample(n=2500, replace=False, random_state=3, axis=0)
        return sample_df

    return (
        # drop missing values
        df.dropna(how="any")
        # convert object to fitting dtype
        .convert_dtypes()
        # remove all strings (no other dtypes are present
        # except for integers and floats)
        .select_dtypes(exclude=["string"])
        # transform all numbers to same dtype
        .astype("float32")
        # drop column with label Agricultural Household indicator
        # because this is not really a numerical input but
        # rather a categorical/classification
        .drop(["Agricultural Household indicator"], axis=1, inplace=False)
        # this column is dropped because it depends on
        # Agricultural Household indicator
        .drop(["Crop Farming and Gardening expenses"], axis=1, inplace=False)
        # use 8 output variables and 24 input variables
        .drop(
            [
                "Total Rice Expenditure",
                "Total Fish and  marine products Expenditure",
                "Fruit Expenditure",
                "Restaurant and hotels Expenditure",
                "Alcoholic Beverages Expenditure",
                "Tobacco Expenditure",
                "Clothing, Footwear and Other Wear Expenditure",
                "Imputed House Rental Value",
                "Transportation Expenditure",
                "Miscellaneous Goods and Services Expenditure",
                "Special Occasions Expenditure",
            ],
            axis=1,
            inplace=False,
        )
        # add input and output labels to each column
        .pipe(add_input_output_family_income)
        # normalize all the columns
        .pipe(normalize_columns_income)
        # remove all datapoints that do not satisfy the constraints
        .pipe(check_constraints_income)
        # sample 2500 examples
        .pipe(sample_2500_examples)
    )


def preprocess_AdultCensusIncome(df: pd.DataFrame) -> pd.DataFrame:  # noqa: N802
    """Preprocesses the Adult Census Income dataset for PyTorch ML.

    Sequential steps:
    - Drop rows with missing values.
    - Encode categorical variables to integer labels.
    - Map the target 'income' column to 0/1.
    - Convert all data to float32.
    - Add a multiindex to denote Input vs Output columns.

    Args:
        df (pd.DataFrame): Raw dataframe containing Adult Census Income data.

    Returns:
        pd.DataFrame: Preprocessed dataframe.
    """

    def drop_missing(df: pd.DataFrame) -> pd.DataFrame:
        """Drop rows with any missing values."""
        return df.dropna(how="any")

    def drop_columns(df: pd.DataFrame) -> pd.DataFrame:
        return df.drop(columns=["fnlwgt", "education.num"], errors="ignore")

    def label_encode_column(series: pd.Series, col_name: str = None) -> pd.Series:
        """Encode a pandas Series of categorical strings into integers."""
        categories = series.dropna().unique().tolist()
        cat_to_int = {cat: i for i, cat in enumerate(categories)}
        if col_name:
            print(f"Column '{col_name}' encoding:")
            for cat, idx in cat_to_int.items():
                print(f"  {cat} -> {idx}")
        return series.map(cat_to_int).astype(int)

    def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
        """Convert categorical string columns to integer labels using label_encode_column."""
        df_temp = df.copy()
        categorical_cols = [
            "workclass",
            "education",
            "marital.status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "native.country",
        ]
        for col in categorical_cols:
            df_temp[col] = label_encode_column(df_temp[col].astype(str), col_name=col)
        return df_temp

    def map_target(df: pd.DataFrame) -> pd.DataFrame:
        """Map income column to 0 (<=50K) and 1 (>50K)."""
        df_temp = df.copy()
        df_temp["income"] = df_temp["income"].map({"<=50K": 0, ">50K": 1})
        return df_temp

    def convert_float32(df: pd.DataFrame) -> pd.DataFrame:
        """Convert all data to float32 for PyTorch compatibility."""
        return df.astype("float32")

    def add_input_output_index(df: pd.DataFrame) -> pd.DataFrame:
        """Add a multiindex indicating input and output columns."""
        temp_df = df.copy()
        column_names = temp_df.columns.tolist()
        # Only the 'income' column is output
        input_list = ["Input"] * (len(column_names) - 1)
        output_list = ["Output"]
        multiindex_list = input_list + output_list
        multiindex = pd.MultiIndex.from_arrays([multiindex_list, column_names])
        return pd.DataFrame(temp_df.to_numpy(), columns=multiindex)

    return (
        df.pipe(drop_missing)
        .pipe(drop_columns)
        .pipe(encode_categorical)
        .pipe(map_target)
        .pipe(convert_float32)
        .pipe(add_input_output_index)
    )
