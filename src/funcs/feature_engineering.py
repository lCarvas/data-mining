from typing import Literal

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


class Preprocessing:
    @staticmethod
    def fillNa(
        data: pd.DataFrame, metricFeatures: list[str], boolFeatures: list[str]
    ) -> pd.DataFrame:
        """Fill missing values

        Args:
            data (`pd.DataFrame`): Dataframe to be treated
            metricFeatures (list[str]): Metric features of the dataset
            boolFeatures (list[str]): Boolean features of the dataset

        Returns:
            `pd.DataFrame`: Treated dataframe
        """
        for var in metricFeatures:
            data[var] = data[var].fillna(data[var].median())

        for var in boolFeatures:
            data[var] = data[var].fillna(0)

        return data

    @staticmethod
    def removeOutliers(
        data: pd.DataFrame,
        thresholds: dict[str, dict[Literal["lower", "upper"], float | None]],
    ) -> pd.DataFrame:
        """Removes outliers and fixes any negative number incoherences on the selected variables from the dataframe

        Args:
            data (`pd.DataFrame`): Dataframe to be treated
            thresholds (dict[str, dict[Literal["lower", "upper"], float  |  None]]): Lower and upper thresholds for each metric feature


        Returns:
            `pd.DataFrame`: Treated dataframe
        """
        for var in thresholds:
            if thresholds[var]["lower"] is not None:
                toRemove: list = list(
                    data.loc[data[var] < thresholds[var]["lower"], var].index
                )
            if thresholds[var]["upper"] is not None:
                toRemove.extend(
                    list(
                        data.loc[
                            data[var] > thresholds[var]["upper"], var
                        ].index
                    )
                )
            data.drop(toRemove, axis=0, inplace=True)

        return data

    @staticmethod
    def encodeCategorial(
        data: pd.DataFrame, catFeatures: list[str]
    ) -> pd.DataFrame:
        """Encode categorical variables with LabelEncoder


        Args:
            data (`pd.DataFrame`): Dataframe to be treated
            catFeatures: list[str]: list of categorical column names that need encoding

        Returns:
            `pd.DataFrame`: Categorical columns are now numeric labels
        """
        le = LabelEncoder()
        for var in catFeatures:
            data[var] = le.fit_transform(data[var].astype(str))
        return data

    @staticmethod
    def scaleFeatures(
        data: pd.DataFrame, metricFeatures: list[str]
    ) -> pd.DataFrame:
        """Standardize numeric features

        Args:
            data (`pd.DataFrame`): Dataframe to be treated
            metricFeatures: list[str]: list of numeric column names

        Returns:
            `pd.DataFrame`:DataFrame where numeric columns are scaled.



        """
        scaler = StandardScaler()
        data[metricFeatures] = scaler.fit_transform(data[metricFeatures])
        return data

    @staticmethod
    def splitdata(
        data: pd.DataFrame,
        target: str,
        test_size: float = 0.2,
        random_state: int = 42,
        stratify: bool = True,
    ):
        x = data.drop(columns=[target])
        y = data[target]
        return train_test_split(
            x,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y if stratify else None,
        )
