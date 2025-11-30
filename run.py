import yfinance as yf
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

#ticker = "^GSPC"    # S&P 500 index (much longer, WARNING: does not include dividends)
#ticker = "^SP500TR"  # S&P 500 index including reinvested dividends (total return)
#ticker = "^GDAXI"   # DAX index including reinvested dividends (total return)
ticker = "data/msci_world_curvo.csv"  # CSV file (needs Date column + price column)
duration_years = 15 # Number of years for each return period
plot_name = "" # Optional; sets custom name in plot title when not empty

scramble_years = True # Pick random years (True: Standard Bootstrap) instead of sequential years (False: Block Bootstrap)
scramble_iteration = 1e5  # Number of iterations for Standard Bootstrap
use_replace = True  # Whether to pick years with replacement (only for Standard Bootstrap): Some debate, but usually done with replacement


plot_name = plot_name if plot_name else ticker
scramble_iteration = int(scramble_iteration)

class StockDataFetcher:
    def __init__(self, ticker="^GSPC", cache_dir="cache"):
        """Initialize the stock data fetcher with caching support.
        
        Parameters:
        -----------
        ticker : str, optional
            Stock ticker symbol or path to CSV file. Default is "^GSPC" (S&P 500 index).
            If ticker ends with .csv, it will be treated as a CSV file path.
        cache_dir : str, optional
            Directory to store cache files. Default is "cache".
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.ticker = ticker
        self.is_csv = ticker.endswith('.csv')
        
    def _get_cache_path(self, start_date, end_date):
        """Generate cache file path based on ticker and date range."""
        # Sanitize ticker for filename (replace special chars)
        safe_ticker = self.ticker.replace("^", "").replace("/", "_")
        filename = f"{safe_ticker}_{start_date}_{end_date}.pkl"
        return self.cache_dir / filename
    
    def _is_cache_valid(self, cache_path, max_age_hours=24):
        """Check if cache file exists and is not too old."""
        if not cache_path.exists():
            return False
        
        # Check file age
        file_time = datetime.fromtimestamp(cache_path.stat().st_mtime)
        age = datetime.now() - file_time
        
        return age < timedelta(hours=max_age_hours)
    
    def _load_from_cache(self, cache_path):
        """Load data from cache file."""
        try:
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            print(f"✓ Loaded data from cache: {cache_path.name}")
            return data
        except Exception as e:
            print(f"✗ Error loading cache: {e}")
            return None
    
    def _save_to_cache(self, data, cache_path):
        """Save data to cache file."""
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
            print(f"✓ Saved data to cache: {cache_path.name}")
        except Exception as e:
            print(f"✗ Error saving cache: {e}")
    
    def _load_from_csv(self, csv_path):
        """Load data from CSV file.
        
        Expected format: CSV with Date column and at least one price column.
        Will try to find columns named: Close, Price, Value, or use the second column.
        """
        import pandas as pd
        
        try:
            # Read CSV file
            df = pd.read_csv(csv_path)
            
            # Find date column (case-insensitive)
            date_col = None
            for col in df.columns:
                if col.lower() in ['date', 'time', 'datetime']:
                    date_col = col
                    break
            
            if date_col is None:
                print(f"✗ No date column found in CSV. Looking for 'Date', 'Time', or 'DateTime'")
                return None
            
            # Convert to datetime and set as index
            df[date_col] = pd.to_datetime(df[date_col])
            df.set_index(date_col, inplace=True)
            df.sort_index(inplace=True)
            
            # Find price column
            price_col = None
            for col in df.columns:
                if col.lower() in ['close', 'price', 'value']:
                    price_col = col
                    break
            
            # If no standard column found, use the first numeric column
            if price_col is None:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    price_col = numeric_cols[0]
                else:
                    print(f"✗ No numeric price column found in CSV")
                    return None
            
            # Create a DataFrame with 'Close' column to match yfinance format
            result = pd.DataFrame(index=df.index)
            result['Close'] = df[price_col]
            
            print(f"✓ Loaded {len(result)} records from CSV")
            print(f"  Using column '{price_col}' as price data")
            
            return result
            
        except Exception as e:
            print(f"✗ Error loading CSV: {e}")
            return None
    
    def fetch_data(self, start_date=None, end_date=None, use_cache=True, max_cache_age_hours=24):
        """
        Fetch historical stock data with caching.
        
        Parameters:
        -----------
        start_date : str, optional
            Start date in 'YYYY-MM-DD' format. Defaults to 10 years ago.
        end_date : str, optional
            End date in 'YYYY-MM-DD' format. Defaults to today.
        use_cache : bool, optional
            Whether to use cached data if available. Default is True.
        max_cache_age_hours : int, optional
            Maximum age of cache in hours. Default is 24 hours.
            
        Returns:
        --------
        pandas.DataFrame
            Historical data with OHLCV (Open, High, Low, Close, Volume) information.
        """
        # Set default dates if not provided
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365*10)).strftime('%Y-%m-%d')
        
        # Handle CSV file
        if self.is_csv:
            csv_path = Path(self.ticker)
            if not csv_path.exists():
                print(f"✗ CSV file not found: {self.ticker}")
                return None
            
            print(f"Loading data from CSV: {self.ticker}")
            data = self._load_from_csv(csv_path)
            
            if data is None:
                return None
            
            # Apply date filtering if specified
            if start_date:
                data = data[data.index >= start_date]
            if end_date:
                data = data[data.index <= end_date]
            
            return data
        
        # Handle ticker symbol (yfinance)
        cache_path = self._get_cache_path(start_date, end_date)
        
        # Try to load from cache
        if use_cache and self._is_cache_valid(cache_path, max_cache_age_hours):
            data = self._load_from_cache(cache_path)
            if data is not None:
                return data
        
        # Fetch from yfinance
        print(f"Fetching {self.ticker} data from {start_date} to {end_date}...")
        try:
            data = yf.download(self.ticker, start=start_date, end=end_date, progress=False)
            
            if data.empty:
                print("✗ No data retrieved from yfinance")
                return None
            
            print(f"✓ Retrieved {len(data)} records from yfinance")
            
            # Save to cache
            if use_cache:
                self._save_to_cache(data, cache_path)
            
            return data
            
        except Exception as e:
            print(f"✗ Error fetching data: {e}")
            return None
    
    def clear_cache(self, ticker_only=True):
        """Clear cached files.
        
        Parameters:
        -----------
        ticker_only : bool, optional
            If True, only clear cache for this ticker. If False, clear all cache.
        """
        count = 0
        if ticker_only:
            safe_ticker = self.ticker.replace("^", "").replace("/", "_")
            pattern = f"{safe_ticker}_*.pkl"
        else:
            pattern = "*.pkl"
        
        for cache_file in self.cache_dir.glob(pattern):
            cache_file.unlink()
            count += 1
        print(f"✓ Cleared {count} cache file(s)")


def get_closest_date(data, target_date):
    """Find the closest available date in the dataset to the target date."""
    idx = data.index.searchsorted(target_date)
    if idx >= len(data.index):
        return data.index[-1]
    elif idx == 0:
        return data.index[0]
    else:
        # Compare which is closer
        before = data.index[idx - 1]
        after = data.index[idx]
        if abs((target_date - before).days) <= abs((after - target_date).days):
            return before
        else:
            return after


def calculate_annual_returns(data):
    """Calculate year-over-year returns from the data using 1-year increments from start date."""
    import pandas as pd
    
    # Handle MultiIndex columns from yfinance
    if isinstance(data.columns, pd.MultiIndex):
        close_col = [col for col in data.columns if col[0] == 'Close'][0]
    else:
        close_col = 'Close'
    
    # Get start and end dates
    start_date = data.index[0]
    end_date = data.index[-1]
    
    # Calculate how many full years we have
    total_years = int((end_date - start_date).days / 365.25)
    
    # Get prices at 1-year increments from start date
    yearly_prices = []
    for year_offset in range(total_years + 1):
        target_date = start_date + timedelta(days=365.25 * year_offset)
        print(target_date)
        actual_date = get_closest_date(data, target_date)
        price = float(data.loc[actual_date, close_col])
        yearly_prices.append(price)
    
    # Calculate year-over-year returns
    annual_returns = []
    for i in range(1, len(yearly_prices)):
        ret = (yearly_prices[i] - yearly_prices[i-1]) / yearly_prices[i-1]
        annual_returns.append(ret)
    
    return annual_returns


def calculate_period_returns(data, duration_years):
    """Calculate returns for rolling periods of specified duration."""
    import pandas as pd
    
    first_date = data.index[0]
    last_date = data.index[-1]
    years_span = (last_date - first_date).days / 365.25
    periods = int(np.floor(years_span) - duration_years)
    
    returns = []
    
    for i in range(periods):
        # Calculate start and end dates for this period
        start_target = first_date + timedelta(days=365.25 * i)
        end_target = start_target + timedelta(days=365.25 * duration_years)
        
        # Find closest available dates
        start_actual = get_closest_date(data, start_target)
        end_actual = get_closest_date(data, end_target)
        
        # Get prices at start and end
        # Handle MultiIndex columns
        if isinstance(data.columns, pd.MultiIndex):
            close_col = [col for col in data.columns if col[0] == 'Close'][0]
        else:
            close_col = 'Close'
        
        start_price = float(data.loc[start_actual, close_col])
        end_price = float(data.loc[end_actual, close_col])
        
        # Calculate actual years between dates
        actual_years = (end_actual - start_actual).days / 365.25
        
        # Calculate total return and annualized return
        total_return = (end_price - start_price) / start_price
        annualized_return = (1 + total_return) ** (1 / actual_years) - 1
        
        returns.append({
            'period': i + 1,
            'start_date': start_actual,
            'end_date': end_actual,
            'start_price': start_price,
            'end_price': end_price,
            'total_return': total_return,
            'return': annualized_return,
            'years': actual_years
        })
    
    return returns


def calculate_scrambled_returns(data, duration_years, iterations):
    """Run multiple iterations by randomly picking duration_years from annual returns."""
    import pandas as pd
    
    print(f"\n=== Running {iterations} scrambled iterations ===")
    
    # Calculate annual returns once
    annual_returns = calculate_annual_returns(data)
    num_years = len(annual_returns)
    
    all_returns = []
    
    for iteration in range(iterations):
        if (iteration + 1) % 1000 == 0:
            print(f"Completed {iteration + 1:,}/{iterations:,} iterations...")
        
        # Randomly pick duration_years returns from annual_returns (each year only once)
        random_indices = np.random.choice(num_years, size=duration_years, replace=use_replace)
        window_returns = [annual_returns[i] for i in random_indices]
        
        # Calculate cumulative return
        cumulative_return = 1.0
        for ret in window_returns:
            cumulative_return *= (1 + ret)
        
        total_return = cumulative_return - 1
        annualized_return = (1 + total_return) ** (1 / duration_years) - 1
        
        all_returns.append({
            'period': len(all_returns) + 1,
            'start_date': None,  # Not meaningful for scrambled data
            'end_date': None,
            'start_price': None,
            'end_price': None,
            'total_return': total_return,
            'return': annualized_return,
            'years': duration_years
        })
    
    print(f"✓ Completed all {iterations} iterations")
    print(f"Total periods analyzed: {len(all_returns)}")
    
    return all_returns


def plot_returns_histogram(returns, duration_years, ticker, best_period, worst_period, overall_return, scrambled=False, first_date=None, last_date=None):
    """Create a histogram of returns with percentile lines."""
    returns_array = np.array([r['return'] * 100 for r in returns])
    
    # Calculate percentiles
    percentiles = {
        '1st': np.percentile(returns_array, 1),
        '10th': np.percentile(returns_array, 10),
        '25th': np.percentile(returns_array, 25),
        '50th (Median)': np.percentile(returns_array, 50),
        '75th': np.percentile(returns_array, 75),
        '90th': np.percentile(returns_array, 90),
        '99th': np.percentile(returns_array, 99)
    }
    
    # Create histogram
    plt.figure(figsize=(14, 8))
    plt.hist(returns_array, bins=30, alpha=0.7, color='steelblue', edgecolor='black', density=True)
    
    # Add percentile lines
    colors = {'1st': 'darkred', '10th': 'red', '25th': 'orange', '50th (Median)': 'green', 
              '75th': 'orange', '90th': 'red', '99th': 'darkred'}
    line_styles = {'1st': ':', '10th': '--', '25th': '--', '50th (Median)': '-', 
                   '75th': '--', '90th': '--', '99th': ':'}
    
    for label, value in percentiles.items():
        plt.axvline(value, color=colors[label], linestyle=line_styles[label], 
                   linewidth=2, label=f'{label}: {value:.1f}% p.a.')
    
    # Add overall period return line
    #plt.axvline(overall_return * 100, color='blue', linestyle=':', 
    #           linewidth=3, label=f'Overall: {overall_return*100:.1f}% p.a.')
    
    plt.xlabel('Annualized Return (% p.a.)', fontsize=12)
    plt.ylabel('Probability Density (%)', fontsize=12)
    
    # Format y-axis as percentage
    ax = plt.gca()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y*100:.1f}'))
    
    method_text = f"{scramble_iteration:_}".replace("_", " ") + " iterations" if scramble_years else "Historical Sequenence"
    date_range = f" data from {first_date.strftime('%Y-%m-%d')} to {last_date.strftime('%Y-%m-%d')}" if first_date and last_date else ""
    plt.title(f'Distribution of Annualized Returns in {duration_years}-Year Windows ({method_text})\n{plot_name}{date_range}', 
              fontsize=14, fontweight='bold')
    plt.legend(loc='upper left', fontsize=10, title='Percentiles', title_fontsize=11)
    plt.grid(axis='y', alpha=0.3)
    
    # Calculate standard deviation
    std_dev = np.std(returns_array)
    
    # Add text box with best and worst periods and standard deviation
    if best_period["start_date"] is not None:
        textstr = (f'Best Period:\n'
                   f'{best_period["start_date"].strftime("%Y-%m-%d")} to {best_period["end_date"].strftime("%Y-%m-%d")}\n'
                   f'Return: {best_period["return"]*100:.2f}% p.a.\n\n'
                   f'Worst Period:\n'
                   f'{worst_period["start_date"].strftime("%Y-%m-%d")} to {worst_period["end_date"].strftime("%Y-%m-%d")}\n'
                   f'Return: {worst_period["return"]*100:.2f}% p.a.\n\n'
                   f'Standard Deviation: {std_dev:.2f}% p.a.')
    else:
        textstr = (f'Best Period:\n'
                   f'Return: {best_period["return"]*100:.2f}% p.a.\n\n'
                   f'Worst Period:\n'
                   f'Return: {worst_period["return"]*100:.2f}% p.a.\n\n'
                   f'Standard Deviation: {std_dev:.2f}% p.a.')
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    plt.text(0.98, 0.97, textstr, transform=plt.gca().transAxes, fontsize=9,
             verticalalignment='top', horizontalalignment='right', bbox=props)
    
    plt.tight_layout()
    plt.savefig(f'returns_histogram_{plot_name.replace("^","").replace("/","_")}_{duration_years}yr.png', dpi=300)
    plt.show()
    
    return percentiles


def main():
    # Fetch all available data for the specified ticker
    print(f"\n=== Fetching {ticker} Data ===")
    fetcher = StockDataFetcher(ticker=ticker)
    # Fetch from a very early date to get all available data
    start_date = "1900-01-01"
    data = fetcher.fetch_data(start_date=start_date)
    
    if data is not None and not data.empty:
        # Calculate how many years the data actually spans
        first_date = data.index[0]
        last_date = data.index[-1]
        years_span = (last_date - first_date).days / 365.25
        
        print(f"\n✓ Data retrieved successfully!")
        print(f"  First date: {first_date.strftime('%Y-%m-%d')}")
        print(f"  Last date: {last_date.strftime('%Y-%m-%d')}")
        print(f"  Years of data: {years_span:.2f} years")
        print(f"  Total records: {len(data)}")
        
        # Check if duration_years is longer than available data
        if duration_years > years_span:
            print(f"\n✗ Error: duration_years ({duration_years}) is longer than available data ({years_span:.2f} years)")
            print(f"   Please reduce duration_years to at most {int(years_span)} years")
            return
        
        # Calculate period returns
        if scramble_years:
            print(f"\n=== Using Scrambled Years Method ===")
            returns = calculate_scrambled_returns(data, duration_years, scramble_iteration)
        else:
            print(f"\n=== Using Sequential Years Method ===")
            returns = calculate_period_returns(data, duration_years)
        
        # Find best and worst periods
        best_period = max(returns, key=lambda x: x['return'])
        worst_period = min(returns, key=lambda x: x['return'])
        
        # Summary statistics
        returns_array = np.array([r['return'] for r in returns])
        print(f"\n=== Summary Statistics ===")
        print(f"Number of periods: {len(returns)}")
        print(f"Average return: {np.mean(returns_array)*100:.2f}%")
        print(f"Median return: {np.median(returns_array)*100:.2f}%")
        print(f"Min return: {np.min(returns_array)*100:.2f}%")
        print(f"Max return: {np.max(returns_array)*100:.2f}%")
        print(f"Std deviation: {np.std(returns_array)*100:.2f}%")
        
        print(f"\n=== Best Period ===")
        if best_period['start_date'] is not None:
            print(f"{best_period['start_date'].strftime('%Y-%m-%d')} to {best_period['end_date'].strftime('%Y-%m-%d')}")
        print(f"Return: {best_period['return']*100:+.2f}%")
        
        print(f"\n=== Worst Period ===")
        if worst_period['start_date'] is not None:
            print(f"{worst_period['start_date'].strftime('%Y-%m-%d')} to {worst_period['end_date'].strftime('%Y-%m-%d')}")
        print(f"Return: {worst_period['return']*100:+.2f}%")
        
        # Calculate overall period return
        overall_start_price = float(data.iloc[0]['Close'])
        overall_end_price = float(data.iloc[-1]['Close'])
        overall_total_return = (overall_end_price - overall_start_price) / overall_start_price
        overall_annualized_return = (1 + overall_total_return) ** (1 / years_span) - 1
        
        print(f"\n=== Overall Period ({years_span:.2f} years) ===")
        print(f"{first_date.strftime('%Y-%m-%d')} to {last_date.strftime('%Y-%m-%d')}")
        print(f"Total return: {overall_total_return*100:+.2f}%")
        print(f"Annualized return: {overall_annualized_return*100:+.2f}%")
        
        # Plot histogram
        plot_returns_histogram(returns, duration_years, ticker, best_period, worst_period, overall_annualized_return, scramble_years, first_date, last_date)
    else:
        print("\n✗ Failed to retrieve data")


if __name__ == "__main__":
    main()
