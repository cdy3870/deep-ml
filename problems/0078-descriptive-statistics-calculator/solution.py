import numpy as np

def descriptive_statistics(data: list | np.ndarray) -> dict:
    """
    Calculate various descriptive statistics metrics for a given dataset.
    
    Args:
        data: List or numpy array of numerical values
    
    Returns:
        Dictionary containing mean, median, mode, variance, standard deviation,
        percentiles (25th, 50th, 75th), and interquartile range (IQR)
    """
    
    data = np.array(data)
    variance = np.square(data - data.mean()).mean()
    std = np.sqrt(variance)
    values, counts = np.unique(data, return_counts=True)
    mode = values[np.argmax(counts)]
    q1, q2, q3 = np.quantile(data, [0.25, 0.5, 0.75])
    # Your code here
    return {"mean": data.mean(), "median": np.median(data), "mode": mode, "variance":  variance, "standard_deviation": std, "25th_percentile": q1, "50th_percentile": q2, "75th_percentile": q3, "interquartile_range": q3 - q1}