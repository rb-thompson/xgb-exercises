# DataBuilder() Class for generating data
import pandas as pd
import numpy as np
from typing import Dict, List, Union, Callable

# This class generates a DataFrame with random data using 
# specified parameters. 
#
# Users can select a category, number of rows/columns, and whether 
# to include missing values. 
#
# Categories include:
# - 'Nature': Generates random data related to nature.
# - 'Finance': Generates random financial data.
# - 'Statistics': Generates random statistical data.
# - 'Machine Learning': Generates random data for machine learning tasks.

class DataBuilder:
    """
    A class for generating random datasets for data science tasks.

    Parameters
    ----------
    category : str, optional (default='Nature')
        The category of data to generate (e.g., 'Nature', 'Finance').
    rows : int, optional (default=10)
        Number of rows in the dataset.
    cols : int, optional (default=None)
        Number of columns in the dataset. If None, uses all columns in the category.
    missing_values : bool, optional (default=False)
        Whether to include missing values in the dataset.
    missing_rate : float or dict, optional (default=0.2)
        Proportion of missing values (float) or column-specific rates (dict).
    custom_config : dict, optional (default=None)
        Custom configuration for data generation.

    Examples
    --------
    >>> builder = DataBuilder(category='Nature', rows=5)
    >>> df = builder.generate_data()
    >>> print(df)
    """
        
    # Default configurations for each category
    DEFAULT_CONFIG = {
        'Nature': {
            'columns': {
                'species': {'type': 'choice', 'values': ['Oak', 'Pine', 'Maple']},
                'height': {'type': 'uniform', 'low': 1.0, 'high': 30.0},
                'age': {'type': 'integers', 'low': 1, 'high': 100}
            }
        },
        'Finance': {
            'columns': {
                'stock': {'type': 'choice', 'values': ['AAPL', 'GOOGL', 'AMZN']},
                'price': {'type': 'uniform', 'low': 100.0, 'high': 1500.0},
                'volume': {'type': 'integers', 'low': 1000, 'high': 1000000}
            }
        },
        'Statistics': {
            'columns': {
                'mean': {'type': 'normal', 'loc': 50, 'scale': 10},
                'std_dev': {'type': 'uniform', 'low': 1, 'high': 20},
                'sample_size': {'type': 'integers', 'low': 30, 'high': 300}
            }
        },
        'Machine Learning': {
            'columns': {
                'feature1': {'type': 'normal', 'loc': 0, 'scale': 1},
                'feature2': {'type': 'normal', 'loc': 0, 'scale': 1},
                'label': {'type': 'choice', 'values': [0, 1]}
            }
        }
    }

    def __init__(self, category: str = 'Nature', rows: int = 10, cols: int = None, 
                missing_values: bool = False, missing_rate: Union[float, Dict] = 0.2, 
                custom_config: Dict = None):
        
        # Validate inputs
        if not isinstance(rows, int) or rows <= 0:
            raise ValueError("Rows must be a positive integer")
        if cols is not None and (not isinstance(cols, int) or cols <= 0):
            raise ValueError("Cols must be a positive integer")
        if not isinstance(missing_values, bool):
            raise ValueError("missing_values must be a boolean")
        
        # Initialize parameters
        self.category = category
        self.rows = rows
        self.cols = cols
        self.missing_values = missing_values
        self.missing_rate = missing_rate
        self.custom_config = custom_config
        self.rng = np.random.default_rng()

        # Validate category
        if self.category not in self.DEFAULT_CONFIG and not self.custom_config:
            raise ValueError(f"Category '{self.category}' not found. Available: {list(self.DEFAULT_CONFIG.keys())}")

        # Use custom config if provided, else default
        self.config = self.custom_config or self.DEFAULT_CONFIG.get(self.category, {})
        # Adjust columns if user specifies cols
        if self.cols is not None:
            self._adjust_columns()

    def _adjust_columns(self):
        """Adjust the number of columns to match user-specified cols."""
        current_cols = list(self.config['columns'].keys())
        if self.cols > len(current_cols):
            # Add generic columns if needed
            for i in range(len(current_cols), self.cols):
                self.config['columns'][f'feature{i+1}'] = {'type': 'normal', 'loc': 0, 'scale': 1}
        elif self.cols < len(current_cols):
            # Trim columns
            self.config['columns'] = {k: v for k, v in list(self.config['columns'].items())[:self.cols]}

    def _generate_column(self, col_config: Dict) -> np.ndarray:
        """Generate data for a single column based on its configuration."""
        col_type = col_config.get('type')
        if col_type == 'choice':
            return self.rng.choice(col_config['values'], size=self.rows)
        elif col_type == 'uniform':
            return self.rng.uniform(col_config['low'], col_config['high'], size=self.rows)
        elif col_type == 'integers':
            return self.rng.integers(col_config['low'], col_config['high'], size=self.rows)
        elif col_type == 'normal':
            return self.rng.normal(col_config.get('loc', 0), col_config.get('scale', 1), size=self.rows)
        else:
            raise ValueError(f"Unsupported column type: {col_type}")
        
    def generate_data(self) -> pd.DataFrame:
        """Generate a DataFrame with random data based on the configuration."""
        data = {}
        for col_name, col_config in self.config['columns'].items():
            data[col_name] = self._generate_column(col_config)

        df = pd.DataFrame(data)

        # Add missing values if specified
        if self.missing_values:
            if isinstance(self.missing_rate, float):
                for col in df.columns:
                    mask = self.rng.choice([True, False], size=self.rows, p=[self.missing_rate, 1-self.missing_rate])
                    df.loc[mask, col] = np.nan
            elif isinstance(self.missing_rate, dict):
                for col, rate in self.missing_rate.items():
                    if col in df.columns:
                        mask = self.rng.choice([True, False], size=self.rows, p=[rate, 1-rate])
                        df.loc[mask, col] = np.nan
        return df
    
    def export_data(self, df: pd.DataFrame, filename: str, format: str = 'csv') -> None:
        """Export DataFrame to specified format."""
        try:
            if format == 'csv':
                df.to_csv(filename, index=False)
            elif format == 'json':
                # Replace NaN with None for JSON compatibility
                df = df.where(pd.notnull(df), None)
                df.to_json(filename, orient='records', lines=True)
            else:
                raise ValueError(f"Unsupported format: {format}")
        except PermissionError:
            raise PermissionError(f"Permission denied when writing to {filename}")
        except OSError as e:
            raise OSError(f"Failed to write to {filename}: {str(e)}")
        
    def add_correlated_feature(self, df: pd.DataFrame, target_col: str, correlation: float = 0.8) -> pd.DataFrame:
        """Add a feature correlated with an existing numeric column."""
        if target_col not in df.columns:
            raise ValueError(f"Column {target_col} not found")
        target = df[target_col].values
        noise = self.rng.normal(0, 1, size=self.rows)
        correlated = correlation * target + np.sqrt(1 - correlation**2) * noise
        df[f'corr_{target_col}'] = correlated
        return df
    
    def list_categories(self) -> List[str]:
        """Return a list of available categories."""
        return list(self.DEFAULT_CONFIG.keys())

    def show_example(self, category: str = None) -> pd.DataFrame:
        """Generate and return an example dataset for the specified or default category."""
        original_category = self.category
        self.category = category or self.category
        df = self.generate_data()
        self.category = original_category
        return df

