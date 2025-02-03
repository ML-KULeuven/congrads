"""
This module provides utility functions and classes for managing data,
logging, and preprocessing in machine learning workflows. The functionalities
include logging key-value pairs to CSV files, splitting datasets into
training, validation, and test sets, preprocessing functions for various 
data sets and validator functions for type checking.
"""

import os

import numpy as np
import pandas as pd
import torch
from torch import Generator
from torch.utils.data import DataLoader, Dataset, random_split


class CSVLogger:
    """
    A utility class for logging key-value pairs to a CSV file, organized by
    epochs. Supports merging with existing logs or overwriting them.

    Args:
        file_path (str): The path to the CSV file for logging.
        overwrite (bool): If True, overwrites any existing file at
            the file_path.
        merge (bool): If True, merges new values with existing data
            in the file.

    Raises:
        ValueError: If both overwrite and merge are True.
        FileExistsError: If the file already exists and neither
            overwrite nor merge is True.
    """

    def __init__(
        self, file_path: str, overwrite: bool = False, merge: bool = True
    ):
        """
        Initializes the CSVLogger.
        """

        self.file_path = file_path
        self.values: dict[tuple[int, str], float] = {}

        if merge and overwrite:
            raise ValueError(
                "The attributes overwrite and merge cannot be True at the "
                "same time. Either specify overwrite=True or merge=True."
            )

        if not os.path.exists(file_path):
            pass
        elif merge:
            self.load()
        elif overwrite:
            pass
        else:
            raise FileExistsError(
                f"A CSV file already exists at {file_path}. Specify "
                "CSVLogger(..., overwrite=True) to overwrite the file."
            )

    def add_value(self, name: str, value: float, epoch: int):
        """
        Adds a value to the logger for a specific epoch and name.

        Args:
            name (str): The name of the metric or value to log.
            value (float): The value to log.
            epoch (int): The epoch associated with the value.
        """

        self.values[epoch, name] = value

    def save(self):
        """
        Saves the logged values to the specified CSV file.

        If the file exists and merge is enabled, merges the current data
        with the existing file.
        """

        data = self.to_dataframe(self.values)
        data.to_csv(self.file_path, index=False)

    def load(self):
        """
        Loads data from the CSV file into the logger.

        Converts the CSV data into the internal dictionary format for
        further updates or operations.
        """

        df = pd.read_csv(self.file_path)
        self.values = self.to_dict(df)

    @staticmethod
    def to_dataframe(values: dict[tuple[int, str], float]) -> pd.DataFrame:
        """
        Converts a dictionary of values into a DataFrame.

        Args:
            values (dict[tuple[int, str], float]): A dictionary of values
                keyed by (epoch, name).

        Returns:
            pd.DataFrame: A DataFrame where epochs are rows, names are
            columns, and values are the cell data.
        """

        # Convert to a DataFrame
        df = pd.DataFrame.from_dict(values, orient="index", columns=["value"])

        # Reset the index to separate epoch and name into columns
        df.index = pd.MultiIndex.from_tuples(df.index, names=["epoch", "name"])
        df = df.reset_index()

        # Pivot the DataFrame so epochs are rows and names are columns
        result = df.pivot(index="epoch", columns="name", values="value")

        # Optional: Reset the column names for a cleaner look
        result = result.reset_index().rename_axis(columns=None)

        return result

    @staticmethod
    def to_dict(df: pd.DataFrame) -> dict[tuple[int, str], float]:
        """
        Converts a DataFrame with epochs as rows and names as columns
        back into a dictionary of the format {(epoch, name): value}.
        """
        # Set the epoch column as the index (if not already)
        df = df.set_index("epoch")

        # Stack the DataFrame to create a multi-index series
        stacked = df.stack()

        # Convert the multi-index series to a dictionary
        result = stacked.to_dict()

        return result


def split_data_loaders(
    data: Dataset,
    loader_args: dict = None,
    train_loader_args: dict = None,
    valid_loader_args: dict = None,
    test_loader_args: dict = None,
    train_size: float = 0.8,
    valid_size: float = 0.1,
    test_size: float = 0.1,
    split_generator: Generator = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Splits a dataset into training, validation, and test sets,
    and returns corresponding DataLoader objects.

    Args:
        data (Dataset): The dataset to be split.
        loader_args (dict, optional): Default DataLoader arguments, merges
            with loader-specific arguments, overlapping keys from
            loader-specific arguments are superseded.
        train_loader_args (dict, optional): Training DataLoader arguments,
            merges with `loader_args`, overriding overlapping keys.
        valid_loader_args (dict, optional): Validation DataLoader arguments,
            merges with `loader_args`, overriding overlapping keys.
        test_loader_args (dict, optional): Test DataLoader arguments,
            merges with `loader_args`, overriding overlapping keys.
        train_size (float, optional): Proportion of data to be used for
            training. Defaults to 0.8.
        valid_size (float, optional): Proportion of data to be used for
            validation. Defaults to 0.1.
        test_size (float, optional): Proportion of data to be used for
            testing. Defaults to 0.1.
        split_generator (Generator, optional): Optional random seed generator
            to control the splitting of the dataset.

    Returns:
        tuple: A tuple containing three DataLoader objects: one for the
        training, validation and test set.

    Raises:
        ValueError: If the train_size, valid_size, and test_size are not
            between 0 and 1, or if their sum does not equal 1.
    """

    # Validate split sizes
    if not (0 < train_size < 1 and 0 < valid_size < 1 and 0 < test_size < 1):
        raise ValueError(
            "train_size, valid_size, and test_size must be between 0 and 1."
        )
    if not abs(train_size + valid_size + test_size - 1.0) < 1e-6:
        raise ValueError("train_size, valid_size, and test_size must sum to 1.")

    # Perform the splits
    train_val_data, test_data = random_split(
        data, [1 - test_size, test_size], generator=split_generator
    )
    train_data, valid_data = random_split(
        train_val_data,
        [
            train_size / (1 - test_size),
            valid_size / (1 - test_size),
        ],
        generator=split_generator,
    )

    # Set default arguments for each loader
    train_loader_args = dict(loader_args or {}, **(train_loader_args or {}))
    valid_loader_args = dict(loader_args or {}, **(valid_loader_args or {}))
    test_loader_args = dict(loader_args or {}, **(test_loader_args or {}))

    # Create the DataLoaders
    train_generator = DataLoader(train_data, **train_loader_args)
    valid_generator = DataLoader(valid_data, **valid_loader_args)
    test_generator = DataLoader(test_data, **test_loader_args)

    return train_generator, valid_generator, test_generator


# pylint: disable-next=invalid-name
def preprocess_BiasCorrection(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses the given dataframe for bias correction by
    performing a series of transformations.

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
        """Transform the string that denotes the date to
        the datetime format in pandas."""
        # make copy of dataframe
        df_temp = df.copy()
        # add new column at the front where the date string is
        # transformed to the datetime format
        df_temp.insert(0, "DateTransformed", pd.to_datetime(df_temp["Date"]))
        return df_temp

    def add_year(df: pd.DataFrame) -> pd.DataFrame:
        """Extract the year from the datetime cell and add it
        as a new column to the dataframe at the front."""
        # make copy of dataframe
        df_temp = df.copy()
        # extract year and add new column at the front containing these numbers
        df_temp.insert(0, "Year", df_temp["DateTransformed"].dt.year)
        return df_temp

    def add_month(df: pd.DataFrame) -> pd.DataFrame:
        """Extract the month from the datetime cell and add it
        as a new column to the dataframe at the front."""
        # make copy of dataframe
        df_temp = df.copy()
        # extract month and add new column at index 1 containing these numbers
        df_temp.insert(1, "Month", df_temp["DateTransformed"].dt.month)
        return df_temp

    def add_day(df: pd.DataFrame) -> pd.DataFrame:
        """Extract the day from the datetime cell and add it
        as a new column to the dataframe at the front."""
        # make copy of dataframe
        df_temp = df.copy()
        # extract day and add new column at index 2 containing these numbers
        df_temp.insert(2, "Day", df_temp["DateTransformed"].dt.day)
        return df_temp

    def add_input_output_temperature(df: pd.DataFrame) -> pd.DataFrame:
        """Add a multiindex denoting if the column
        is an input or output variable."""
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
        multiindex_bias = pd.MultiIndex.from_arrays(
            [input_output_list, column_names]
        )
        # transpose such that index can be adjusted to multi index
        new_df = pd.DataFrame(df.transpose().to_numpy(), index=multiindex_bias)
        # transpose back such that columns are the same as before
        # except with different labels
        return new_df.transpose()

    def normalize_columns_bias(df: pd.DataFrame) -> pd.DataFrame:
        """Normalize the columns for the bias correction dataset.
        This is different from normalizing all the columns separately
        because the upper and lower bounds for the output variables
        are assumed to be the same."""
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
            temp_df[feature_name] = (df[feature_name] - min_value) / (
                max_value - min_value
            )
        return temp_df

    def sample_2500_examples(df: pd.DataFrame) -> pd.DataFrame:
        """Sample 2500 examples from the dataframe without replacement."""
        temp_df = df.copy()
        sample_df = temp_df.sample(
            n=2500, replace=False, random_state=3, axis=0
        )
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


# pylint: disable-next=invalid-name
def preprocess_FamilyIncome(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses the given Family Income dataframe by applying a
    series of transformations and constraints.

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
        """Normalize the columns for the Family Income dataframe.
        This can also be applied to other dataframes because this
        function normalizes all columns individually."""
        # copy the dataframe
        temp_df = df.copy()
        # normalize each column
        for feature_name in df.columns:
            max_value = df[feature_name].max()
            min_value = df[feature_name].min()
            temp_df[feature_name] = (df[feature_name] - min_value) / (
                max_value - min_value
            )
        return temp_df

    def check_constraints_income(df: pd.DataFrame) -> pd.DataFrame:
        """Check if all the constraints are satisfied for the dataframe
        and remove the examples that do not satisfy the constraint.
        This function only works for the Family Income dataset and the
        constraints are that the household income is larger than all the
        expenses and the food expense is larger than the sum of
        the other (more detailed) food expenses.
        """
        temp_df = df.copy()
        # check that household income is larger than expenses in the output
        input_array = temp_df["Input"].to_numpy()
        income_array = np.add(
            np.multiply(
                input_array[:, [0, 1]],
                np.subtract(
                    np.asarray([11815988, 9234485]), np.asarray([11285, 0])
                ),
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
        sanity_check_array = np.greater_equal(
            food_expense_array, food_mul_expense_array_sum
        )
        drop_reduction["Unimportant"] = sanity_check_array.tolist()
        new_reduction = drop_reduction[drop_reduction.Unimportant]
        satisfied_constraints_df = new_reduction.drop("Unimportant", axis=1)

        return satisfied_constraints_df

    def add_input_output_family_income(df: pd.DataFrame) -> pd.DataFrame:
        """Add a multiindex denoting if the column is
        an input or output variable."""

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
        multiindex_bias = pd.MultiIndex.from_arrays(
            [input_output_list, column_names]
        )
        # transpose such that index can be adjusted to multi index
        new_df = pd.DataFrame(df.transpose().to_numpy(), index=multiindex_bias)
        # transpose back such that columns are the same as
        # before except with different labels
        return new_df.transpose()

    def sample_2500_examples(df: pd.DataFrame) -> pd.DataFrame:
        """Sample 2500 examples from the dataframe without replacement."""
        temp_df = df.copy()
        sample_df = temp_df.sample(
            n=2500, replace=False, random_state=3, axis=0
        )
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


def validate_type(name, value, expected_types, allow_none=False):
    """
    Validate that a value is of the specified type(s).

    Args:
        name (str): Name of the argument for error messages.
        value: Value to validate.
        expected_types (type or tuple of types): Expected type(s) for the value.
        allow_none (bool): Whether to allow the value to be None.
            Defaults to False.

    Raises:
        TypeError: If the value is not of the expected type(s).
    """

    if value is None:
        if not allow_none:
            raise TypeError(f"Argument {name} cannot be None.")
        return

    if not isinstance(value, expected_types):
        raise TypeError(
            f"Argument {name} '{str(value)}' is not supported. "
            f"Only values of type {str(expected_types)} are allowed."
        )


def validate_iterable(
    name,
    value,
    expected_element_types,
    allowed_iterables=(list, set),
    allow_none=False,
):
    """
    Validate that a value is an iterable (e.g., list, set) with elements of
    the specified type(s).

    Args:
        name (str): Name of the argument for error messages.
        value: Value to validate.
        expected_element_types (type or tuple of types): Expected type(s)
            for the elements.
        allowed_iterables (tuple of types): Iterable types that are
            allowed (default: list and set).
        allow_none (bool): Whether to allow the value to be None.
            Defaults to False.

    Raises:
        TypeError: If the value is not an allowed iterable type or if
            any element is not of the expected type(s).
    """

    if value is None:
        if not allow_none:
            raise TypeError(f"Argument {name} cannot be None.")
        return

    if not isinstance(value, allowed_iterables):
        raise TypeError(
            f"Argument {name} '{str(value)}' is not supported. "
            f"Only values of type {str(allowed_iterables)} are allowed."
        )
    if not all(
        isinstance(element, expected_element_types) for element in value
    ):
        raise TypeError(
            f"Invalid elements in {name} '{str(value)}'. "
            f"Only elements of type {str(expected_element_types)} are allowed."
        )


def validate_comparator_pytorch(name, value):
    """
    Validate that a value is a callable PyTorch comparator function.

    Args:
        name (str): Name of the argument for error messages.
        value: Value to validate.

    Raises:
        TypeError: If the value is not callable or not a PyTorch comparator.
    """

    # List of valid PyTorch comparator functions
    pytorch_comparators = {torch.gt, torch.lt, torch.ge, torch.le}

    # Check if value is callable and if it's one of
    # the PyTorch comparator functions
    if not callable(value):
        raise TypeError(
            f"Argument {name} '{str(value)}' is not supported. "
            "Only callable functions are allowed."
        )

    if value not in pytorch_comparators:
        raise TypeError(
            f"Argument {name} '{str(value)}' is not a valid PyTorch comparator "
            "function. Only PyTorch functions like torch.gt, torch.lt, "
            "torch.ge, torch.le are allowed."
        )


def validate_callable(name, value, allow_none=False):
    """
    Validate that a value is callable function.

    Args:
        name (str): Name of the argument for error messages.
        value: Value to validate.
        allow_none (bool): Whether to allow the value to be None.
            Defaults to False.

    Raises:
        TypeError: If the value is not callable.
    """

    if value is None:
        if not allow_none:
            raise TypeError(f"Argument {name} cannot be None.")
        return

    if not callable(value):
        raise TypeError(
            f"Argument {name} '{str(value)}' is not supported. "
            "Only callable functions are allowed."
        )


def validate_loaders() -> None:
    """
    TODO: implement function
    """
    # TODO complete
    pass
