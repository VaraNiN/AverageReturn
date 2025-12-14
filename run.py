import yfinance as yf
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from multiprocessing import Pool, Manager
import os
import psutil
import time

# ===================== USER SETTINGS =====================
# ticker accepts either a stock ticker symbol (for yfinance) or a path to a CSV file
# .csv files in the data directory were dowloaded from https://curvo.eu/backtest/en and https://www.macrotrends.net/1333/historical-gold-prices-100-year-chart 
# .pkl files in the cache directory were downloaded via yfinance

#ticker = "^GSPC"    # S&P 500 index (much longer, WARNING: does not include dividends)
#ticker = "^SP500TR"  # S&P 500 index including reinvested dividends (total return)
#ticker = "^GDAXI"   # DAX index including reinvested dividends (total return)
ticker = "data/msci_world_curvo.csv"  # CSV file (needs Date column + price column)
duration_years = 15 # Number of years for each return period
bin_width = 1.0  # Width of histogram bins in percentage points (e.g., 1.0 = 1% bins)
plot_name = "MSCI World" # Optional; sets custom name in plot title when not empty

block_bootstrap = True # True: Standard Bootstrap (sample randomly); False: Block Bootstrap (sequential years)
bootstrap_samples = 5e6  # Number of samples for Standard Bootstrap (5e8 is approximate maximum for 32GB RAM)
use_replace = True  # Whether to pick years with replacement (only for Standard Bootstrap): Some debate, but usually done with replacement

percentiles = [1, 10, 25, 50, 75, 90, 99]  # Percentiles to show as lines on the histogram (e.g., [1, 10, 50, 90, 99])
probability_thresholds = [0.0, 2.5, 7.0, 10.0, 15.0, 20.0]  # Show probability of returns ≥ these values (in % p.a.)
probability_losses = [0, 10, 20, 33, 50, 67, 80, 90]  # Show probability of losses after specified years (in % of initial investment)

random_seed = None  # Random seed for reproducibility (None for random)
num_threads = None  # Number of threads to use for parallel processing (None = use all available CPUs)
chunk_size = 1e5    # Uses chunking for performance optimization; WARNING: DON'T SET THIS TOO LARGE! My system crashed at 1e7.

# =================== END USER SETTINGS ===================

# ============ Disclaimer / Haftungsausschluss ============
# I make no warranty as to the accuracy of the data shown. It's best to verify the calculations yourself.
# Use of this calculator is at your own risk.
# Investments involve risks, including the risk of capital loss.
# This calculator does not constitute financial advice and I am not a financial advisor.
# I accept no liability for losses or damages arising from the use of this program.
# Generative AI tools (Claude Sonnet 4.5) were used in the creation of this program.

# Ich übernehme keine Gewähr auf die Richtigkeit der gezeigten Daten. Am besten selber nochmal nachrechnen.
# Die Nutzung dieses Rechners folgt auf eigene Verantwortung.
# Investitionen sind mit Risiken verbunden, einschließlich des Risikos von Kapitalverlusten.
# Dieser Rechner stellt keine Finanzberatung dar und ich bin kein Finanzberater.
# Ich übernehme keine Haftung für Verluste oder Schäden, die aus der Nutzung dieses Programms entstehen.
# Bei der Erstellung dieses Programms wurden generative KI-Tools (Claude Sonnet 4.5) verwendet.
# ========================================================


plot_name = plot_name if plot_name else ticker
bootstrap_samples = int(bootstrap_samples)
chunk_size = int(chunk_size)
if random_seed is not None:
    np.random.seed(random_seed)

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


def _worker_sample_returns(args):
    """Worker function for parallel sampling of returns (vectorized in chunks).
    Returns only the annualized returns as a 1D array to minimize memory transfer."""
    annual_returns, num_years, duration_years, use_replace, n_samples, seed_offset, progress_queue = args
    
    # Set random seed for this worker 
    if random_seed is not None:
        # Use a combination of base seed, worker ID, and large prime to ensure different sequences
        np.random.seed((random_seed + seed_offset * 982451653) % (2**32))
    else:
        np.random.seed(None)
    
    # Convert to numpy array for faster indexing
    annual_returns_array = np.array(annual_returns)
    
    # Process in chunks and only keep annualized returns
    all_annualized_returns = np.empty(n_samples, dtype=np.float64)
    
    for chunk_start in range(0, n_samples, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_samples)
        chunk_n_samples = chunk_end - chunk_start
        
        # Generate random indices for this chunk
        random_indices = np.random.choice(num_years, size=(chunk_n_samples, duration_years), replace=use_replace)
        
        # Get all window returns for this chunk
        window_returns = annual_returns_array[random_indices]  # Shape: (chunk_n_samples, duration_years)
        
        # Calculate cumulative returns for all samples in this chunk
        cumulative_returns = np.prod(1 + window_returns, axis=1)  # Shape: (chunk_n_samples,)
        
        # Calculate annualized returns and store in pre-allocated array
        all_annualized_returns[chunk_start:chunk_end] = cumulative_returns ** (1 / duration_years) - 1
        
        # Update progress for this chunk
        if progress_queue is not None:
            progress_queue.put(chunk_n_samples)
    
    # Return only the annualized returns array (much smaller than full dict structure)
    return all_annualized_returns


def calculate_scrambled_returns(data, duration_years, bootstrap_samples):
    """Run multiple samples by randomly picking duration_years from annual returns."""
    
    print(f"\n=== Picking {bootstrap_samples:,} samples ===")
    
    # Calculate annual returns once
    annual_returns = calculate_annual_returns(data)
    num_years = len(annual_returns)
    
    # Determine number of threads
    n_threads = num_threads if num_threads is not None else os.cpu_count()
    print(f"Using {n_threads}/{os.cpu_count()} threads for parallel processing")
    
    # Split work among threads
    samples_per_thread = bootstrap_samples // n_threads
    remainder = bootstrap_samples % n_threads
    
    # Create shared progress queue
    with Manager() as manager:
        progress_queue = manager.Queue()
        
        # Prepare arguments for each worker
        worker_args = []
        for i in range(n_threads):
            n_samples = samples_per_thread + (1 if i < remainder else 0)
            worker_args.append((annual_returns, num_years, duration_years, use_replace, n_samples, i, progress_queue))
        
        # Run workers in parallel with progress monitoring
        start_time = time.time()
        completed = 0
        bar_length = 40
        
        with Pool(processes=n_threads) as pool:
            # Start async processing
            async_result = pool.map_async(_worker_sample_returns, worker_args)
            
            # Monitor progress while workers are running
            while not async_result.ready():
                # Check for progress updates
                while not progress_queue.empty():
                    try:
                        completed += progress_queue.get_nowait()
                    except:
                        break
                
                percent = (completed / bootstrap_samples) * 100
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                eta = (bootstrap_samples - completed) / rate if rate > 0 else 0
                
                # Progress bar
                filled = int(bar_length * completed / bootstrap_samples)
                bar = '█' * filled + '░' * (bar_length - filled)
                
                print(f'\r[{bar}] {percent:.1f}% ({completed:,}/{bootstrap_samples:,}) - '
                      f'{rate:.0f} samples/s - ETA: {eta:.0f}s', end='', flush=True)
                
                time.sleep(0.1)
            
            # Get results (list of numpy arrays, one per worker)
            worker_return_arrays = async_result.get()
            
            # Process any remaining progress updates
            while not progress_queue.empty():
                try:
                    completed += progress_queue.get_nowait()
                except:
                    break
            
            # Final progress update
            elapsed = time.time() - start_time
            print(f'\r[{"█" * bar_length}] 100.0% ({bootstrap_samples:,}/{bootstrap_samples:,}) - '
                  f'Completed in {elapsed:.1f}s ({int(bootstrap_samples/elapsed):,} samples/s)')
        
        # Concatenate all worker results into single array
        all_annualized_returns = np.concatenate(worker_return_arrays)
    
    print(f"Total periods analyzed: {len(all_annualized_returns):,}")
    
    # Return numpy array directly instead of converting to dicts
    # This saves huge amount of memory for large sample sizes
    return all_annualized_returns


def plot_returns_histogram(returns_array, duration_years, ticker, best_period, worst_period, overall_return, scrambled=False, first_date=None, last_date=None):
    """Create a histogram of returns with percentile lines.
    
    Parameters:
    -----------
    returns_array : numpy.ndarray
        Array of annualized returns (not percentages)
    """
    print(f"\n=== Plotting Histogram ===")
    # Convert to percentage
    returns_array = returns_array * 100
    n = len(returns_array)
    # Sort in advance for later calculations
    print("Sorting returns...")
    
    # Check available memory before attempting parallel sort
    available_memory = psutil.virtual_memory().available
    array_size = returns_array.nbytes
    n_chunks = num_threads if num_threads is not None else os.cpu_count()
    
    estimated_memory_needed = array_size * 3 # Estimate memory needed for parallel sort
    use_max_available = 0.9  # Use up to 90% of available memory
    
    if estimated_memory_needed < available_memory * use_max_available:
        try:
            print(f"Available memory: {available_memory / 1e9:.1f} GB, estimated need: {estimated_memory_needed / 1e9:.1f} GB")
            print("Using parallel sorting...")
            # Split array into chunks and sort in parallel
            chunk_size = n // n_chunks
            chunks = [returns_array[i*chunk_size:(i+1)*chunk_size] for i in range(n_chunks-1)]
            chunks.append(returns_array[(n_chunks-1)*chunk_size:])  # Last chunk gets remainder
            
            with Pool(processes=num_threads) as pool:
                sorted_chunks = pool.map(np.sort, chunks)
            del chunks  # Free memory after pool processing
            # Merge sorted chunks
            print("Merging sorted chunks...")
            sorted_returns = np.concatenate(sorted_chunks)
            del sorted_chunks  # Free memory before final sort
            sorted_returns.sort(kind='mergesort')  # Final merge sort is faster on partially sorted data
            print("Parallel sorting completed.")
        except Exception as e:
            print(f"Parallel sorting failed ({e}), falling back to standard sort...")
            sorted_returns = np.sort(returns_array)
    else:
        try:
            print(f"Insufficient memory for parallel sort. Available: {available_memory / 1e9:.1f} GB, needed: {estimated_memory_needed * (2 - use_max_available) / 1e9:.1f} GB")
            print("Using single-threaded sort...")
            sorted_returns = np.sort(returns_array) # Have to use slower sorting for extremely large arrays due to memory constraints
        except Exception as e:
            print(f"Sorting failed: {e}")
            exit(1)
    
    # Calculate percentiles using sorted array
    percentile_values = {}
    for p in percentiles:
        if p == 1:
            label = '1st'
        elif p == 2:
            label = '2nd'
        else:
            label = f'{p}th' if p != 50 else '50th (Median)'
        percentile_values[label] = sorted_returns[int(n * p / 100)]
    
    # Calculate plot limits based on 0.01st and 99.99th percentiles
    lower_limit = sorted_returns[int(n * 0.0001)] - 1
    upper_limit = sorted_returns[int(n * 0.9999)] + 1
    
    plt.figure(figsize=(14, 8))
    bins = np.arange(np.floor(sorted_returns[0]) - bin_width / 2., np.ceil(sorted_returns[-1]) + bin_width / 2., bin_width)
    
    # Precompute histogram with numpy (much faster than letting matplotlib do it)
    print("Computing histogram...")
    hist_counts, bin_edges = np.histogram(sorted_returns, bins=bins, density=True)
    
    # Plot using bar chart instead of hist (since we already computed the histogram)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    plt.bar(bin_centers, hist_counts, width=bin_width, alpha=0.7, color='steelblue', edgecolor='black')
    
    # Set x-axis limits to 0.01st and 99.99th percentiles +/- 1
    plt.xlim(lower_limit, upper_limit)
    
    # Add percentile lines with dynamic colors based on extremity
    for label, value in percentile_values.items():
        # Extract numeric percentile value from label (handles '1st', '2nd', '50th (Median)', etc.)
        p_value = int(label.split('st')[0]) if 'st' in label else int(label.split('nd')[0]) if 'nd' in label else int(label.split('th')[0])
        
        # Determine color and line style based on percentile value
        if p_value <= 1 or p_value >= 99:
            color = 'darkred'
            linestyle = ':'
        elif p_value <= 10 or p_value >= 90:
            color = 'red'
            linestyle = '--'
        elif p_value <= 25 or p_value >= 75:
            color = 'orange'
            linestyle = '--'
        elif p_value == 50:
            color = 'green'
            linestyle = '-'
        else:
            color = 'blue'
            linestyle = '--'
        
        plt.axvline(value, color=color, linestyle=linestyle, 
                   linewidth=2, label=f'{label}: {value:.2f}% p.a.')
    
    # Add overall period return line
    #plt.axvline(overall_return * 100, color='blue', linestyle=':', 
    #           linewidth=3, label=f'Overall: {overall_return*100:.1f}% p.a.')
    
    plt.xlabel('Annualized Nominal Return (% p.a.)', fontsize=12)
    plt.ylabel('Probability (%)', fontsize=12)
    
    # Format y-axis as percentage (density * bin_width gives probability)
    ax = plt.gca()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y*bin_width*100:.1f}'))
    
    method_text = f"{bootstrap_samples:_}".replace("_", " ") + " samples" if block_bootstrap else "Historical Sequenence"
    date_range = f" data from {first_date.strftime('%Y-%m-%d')} to {last_date.strftime('%Y-%m-%d')}" if first_date and last_date else ""
    plt.title(f'Distribution of Annualized Returns in {duration_years}-Year Windows ({method_text})\n{plot_name}{date_range}', 
              fontsize=14, fontweight='bold')
    plt.legend(loc='upper left', fontsize=10, title='Percentiles', title_fontsize=11)
    plt.grid(axis='y', alpha=0.3)
    
    # Calculate standard deviation and probability thresholds
    std_dev = np.std(returns_array)
    skewness = (np.mean((returns_array - np.mean(returns_array))**3)) / (std_dev**3)
    kurtosis = (np.mean((returns_array - np.mean(returns_array))**4)) / (std_dev**4) - 3  # Excess kurtosis
    
    # Calculate probabilities for user-defined thresholds
    prob_lines = ["\nProbabilities for:\n"]

    for threshold in probability_thresholds:
        prob = np.sum(returns_array >= threshold) / len(returns_array) * 100
        if prob > 0.1:
          prob_lines.append(f'At least {threshold}%p.a.: {prob:.2f}%')
        else:
          count = np.sum(returns_array >= threshold)
          if count == 0:
            continue
          else:
            one_in_x = len(returns_array) / count
            prob_lines.append(f'At least {threshold}%p.a.: 1 in {one_in_x:,.0f}')

    prob_lines.append('')
    for loss in probability_losses:
        threshold = ((1 - loss/100)**(1/duration_years) - 1) * 100
        count = np.sum(returns_array <= threshold)
        prob = count / len(returns_array) * 100
        if loss != 0:
          if prob > 0.1:
            prob_lines.append(f'At least {loss:.0f}% loss: {prob:.2f}%')
          else:
            if count == 0:
              continue
            else:
              one_in_x = len(returns_array) / count
              prob_lines.append(f'At least {loss:.0f}% loss: 1 in {one_in_x:,.0f}')
        else:
          prob_lines.append(f'No loss: {(100-prob):.2f}%')
          prob_lines.append(f'Any loss: {(prob):.2f}%')
    
    # Add text box with best and worst periods and standard deviation
    if best_period["start_date"] is not None:
        textstr = (f'Best Period:\n'
                   f'{best_period["start_date"].strftime("%Y-%m-%d")} to {best_period["end_date"].strftime("%Y-%m-%d")}\n'
                   f'Return: {best_period["return"]*100:.2f}% p.a.\n\n'
                   f'Worst Period:\n'
                   f'{worst_period["start_date"].strftime("%Y-%m-%d")} to {worst_period["end_date"].strftime("%Y-%m-%d")}\n'
                   f'Return: {worst_period["return"]*100:.2f}% p.a.\n\n'
                   f'Standard Deviation: {std_dev:.2f}% p.a.\n'
                   f'Skewness: {skewness:.3f}\n'
                   f'Excess Kurtosis: {kurtosis:.3f}\n\n' +
                   '\n'.join(prob_lines))
    else:
        textstr = (f'Best Period:\n'
                   f'Return: {best_period["return"]*100:.2f}% p.a.\n\n'
                   f'Worst Period:\n'
                   f'Return: {worst_period["return"]*100:.2f}% p.a.\n\n'
                   f'Standard Deviation: {std_dev:.2f}% p.a.\n'
                   f'Skewness: {skewness:.3f}\n'
                   f'Excess Kurtosis: {kurtosis:.3f}\n\n' +
                   '\n'.join(prob_lines))
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    plt.text(0.98, 0.97, textstr, transform=plt.gca().transAxes, fontsize=11,
             verticalalignment='top', horizontalalignment='right', bbox=props)
    
    plt.tight_layout()
    plt.savefig(f'returns_histogram_{plot_name.replace("^","").replace("/","_")}_{duration_years}yr.png', dpi=300)
    
    return percentiles


def show_plot_with_timer(start_time):
    """Show plot and print total execution time."""
    total_time = time.time() - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = total_time % 60
    print(f"\n{'='*60}")
    print(f"Total program execution time: {hours} hours, {minutes} minutes, {seconds:.3f} seconds")
    print(f"{'='*60}")
    plt.show()


def main():
    # Start timer for total program execution
    program_start_time = time.time()
    
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
        if block_bootstrap:
            print(f"\n=== Using Standard Bootstrap Method ===")
            returns_array = calculate_scrambled_returns(data, duration_years, bootstrap_samples)
        else:
            print(f"\n=== Using Moving Block Bootstrap Method ===")
            returns_list = calculate_period_returns(data, duration_years)
            # Convert to numpy array for consistency
            returns_array = np.array([r['return'] for r in returns_list])
        
        # Summary statistics (work directly with numpy array)
        print(f"\n=== Summary Statistics ===")
        print(f"Number of periods: {len(returns_array):,}")
        print(f"Average return: {np.mean(returns_array)*100:.2f}%")
        print(f"Median return: {np.median(returns_array)*100:.2f}%")
        print(f"Min return: {np.min(returns_array)*100:.2f}%")
        print(f"Max return: {np.max(returns_array)*100:.2f}%")
        print(f"Std deviation: {np.std(returns_array)*100:.2f}%")
        
        # Find best and worst periods (create minimal dicts for display)
        max_idx = np.argmax(returns_array)
        min_idx = np.argmin(returns_array)
        
        best_period = {
            'start_date': None,
            'end_date': None,
            'return': float(returns_array[max_idx]),
            'years': duration_years
        }
        worst_period = {
            'start_date': None,
            'end_date': None,
            'return': float(returns_array[min_idx]),
            'years': duration_years
        }
        
        print(f"\n=== Best Period ===")
        print(f"Return: {best_period['return']*100:+.2f}%")
        
        print(f"\n=== Worst Period ===")
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
        
        # Plot histogram (pass numpy array directly)
        plot_returns_histogram(returns_array, duration_years, ticker, best_period, worst_period, overall_annualized_return, block_bootstrap, first_date, last_date)
        
        # Show plot with execution time
        show_plot_with_timer(program_start_time)
    else:
        print("\n✗ Failed to retrieve data")


if __name__ == "__main__":
    main()
