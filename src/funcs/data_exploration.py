import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from IPython.display import Markdown, display

sns.set_theme()


class DataExploration:
    @staticmethod
    def import_data(path: str = "") -> pd.DataFrame:
        """Imports the data from the specified path and outputs it as a dataframe

        Args:
            path (str): path from which to import the data

        Returns:
            pd.DataFrame: Dataframe
        """
        dataCustomer: pd.DataFrame = pd.read_csv(
            path + "data/DM_AIAI_CustomerDB.csv"
        )
        dataFlights: pd.DataFrame = pd.read_csv(
            path + "data/DM_AIAI_FlightsDB.csv"
        )

        dataCustomer = dataCustomer.set_index("Loyalty#")
        dataFlights = dataFlights.set_index("Loyalty#")

        dataFinal = dataCustomer.join(dataFlights, "Loyalty#")

        return dataFinal

    @staticmethod
    def describeData(
        data: pd.DataFrame,
        metricFeatures: list[str],
        categoricalFeatures: list[str],
    ) -> pd.DataFrame:
        """Outputs information about the dataframe

        Args:
            data (pd.DataFrame): Dataframe to be analyzed
            metricFeatures (list[str]): Metric features of the dataframe
            categoricalFeatures (list[str]): Categorical features of the dataframe

        Returns:
            pd.DataFrame: Describe function output
        """
        print(
            f"Duplicaded: {data.duplicated().sum()}\nMissing: {data.isna().sum().sum()}\nNon-Registered (empty): {(data['Registered'] != 'Yes').sum()}"
        )
        display(Markdown("### Value Counts"))
        for variable in categoricalFeatures:
            print(data[variable].value_counts())

        display(Markdown("\n\n### Percentage of missing values per column"))
        print(round(data.isnull().sum() / data.shape[0] * 100.00, 2))

        # region problem for later
        # data = data.drop("Observations", axis=1).drop_duplicates()
        # data = data[data["Registered"] == "Yes"]
        # data = data.drop("Registered", axis=1)
        # endregion

        for i, col in enumerate(metricFeatures):
            plt.figure(i)
            sns.boxplot(x=col, data=data)

        return data.describe(include="all")
    
    
    @staticmethod
    def correlation_matrix(data: pd.DataFrame, metricFeatures: list[str]):
        """
        Visualizing Correlations between numeric features
        
        
        Args:
            data (pd.DataFrame): Dataframe to be analyzed
            metricFeatures (list[str]): Metric features of the dataframe
        """
    
        correlation = data[metricFeatures].corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation, annot=True, fmt=".2f", cmap="columns")
        plt.title("correlation matrix")
        plt.show()
