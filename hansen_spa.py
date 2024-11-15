import numpy as np
import pandas as pd
from scipy.stats import norm

def hansen_spa(log_returns_df: pd.DataFrame, risk_free_rate: float=0.0,
               null_sharpe: float=0.0, alpha: float=0.05) -> list[str]:
    """
    Perform Hansen's Superior Predictive Ability (SPA) test on a set of strategies.

    The SPA test evaluates the predictive performance of multiple strategies by comparing their
    Sharpe ratios against a specified null Sharpe ratio, accounting for correlation between strategies.
    It adjusts for multiple comparisons and identifies strategies with statistically significant out-performance.

    Args:
    log_returns_df (pd.DataFrame) : DataFrame of log returns, where each column represents a strategy and
    each row represents a time period (index should be timestamps or periods).

    risk_free_rate (float, optional) : The risk-free rate to compute excess returns.
    Default is 0, assuming returns are already excess returns.

    null_sharpe (float, optional) : The hypothesized Sharpe ratio under the null hypothesis.
    Default is 0, testing for performance above zero.

    alpha (float, optional) : The significance level for the test, default is 0.05.

    Returns:
    list[str] : A list of strategy names (column names from `log_returns_df`) that are statistically
    significant at the given alpha level. If no strategies meet the significance threshold, an empty list is returned.

    Notes:
    This function implements Hansen's SPA test by:
    1. Calculating excess Sharpe ratios.
    2. Transforming Sharpe ratios to account for correlation using
    Equation 15 from Pav, S. E. (2019) "Conditional inference on the asset with maximum Sharpe ratio".
    3. Applying a threshold to identify high-performing strategies.
    4. Adjusting for multiple comparisons and identifying statistically significant strategies.

    References:
    - Pav, S. E. (2019). "Conditional inference on the asset with maximum Sharpe ratio".
    - Hansen, P. R. (2005). "A Test for Superior Predictive Ability". Journal of Business & Economic Statistics.
    """
    # Calculate excess returns
    excess_returns: pd.DataFrame = log_returns_df - risk_free_rate
    mean_returns: pd.Series = excess_returns.mean(axis=0)
    std_devs: pd.Series = excess_returns.std(axis=0, ddof=1)
    sharpe_ratios: pd.Series = mean_returns / std_devs

    # Transform the array of sharpe_ratios to array xi
    correlation_matrix: np.ndarray = np.corrcoef(log_returns_df.values, rowvar=False)
    zeta_bar: np.floating = np.mean(sharpe_ratios)
    upper_triangle_indices: tuple = np.triu_indices_from(correlation_matrix, k=1)
    rho: float = float(np.mean(correlation_matrix[upper_triangle_indices]))
    c: float = 1 / np.sqrt(1 + (len(sharpe_ratios) - 1) * rho)
    term1: np.ndarray = c * zeta_bar * np.ones_like(sharpe_ratios)
    term2: pd.Series = (1 / np.sqrt(1 - rho)) * (sharpe_ratios - zeta_bar)
    xi: pd.Series = term1 + term2

    # Rejection threshold
    n: int = len(log_returns_df)
    threshold: float = c * null_sharpe - np.sqrt(2 * np.log(np.log(n)) / n)

    # Calculate k_tilde (number of elements in xi greater than threshold)
    k_tilde: int = np.sum(xi > threshold)
    if k_tilde == 0:
        print('All sharpe ratios evaluated are not statistically significant.')
        return []

    # Calculate critical value and test the null hypothesis
    deviation: pd.Series = xi - c * null_sharpe
    critical_value: float = norm.ppf(1 - alpha / k_tilde)
    statistically_significant: list[str] = deviation[deviation > critical_value].index.to_list()

    if len(statistically_significant) > 0:
        print(f'Statistically significant strategies: {statistically_significant}')
    else:
        print('All sharpe ratios evaluated are not statistically significant.')
    return statistically_significant

if __name__=='__main__':
    # Suppose log_returns_df is your DataFrame of log returns with each column representing a strategy
    log_returns: pd.DataFrame = pd.read_csv(r'log_returns_matrix.csv', index_col='timestamp')
    statistically_significant_results: list[str] = hansen_spa(log_returns, alpha=0.05)
