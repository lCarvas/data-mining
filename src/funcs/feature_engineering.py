from typing import Literal

import pandas as pd


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
