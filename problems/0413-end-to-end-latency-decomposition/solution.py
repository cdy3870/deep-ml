import numpy as np

def decompose_latency(stage_latencies: dict, percentiles: list) -> dict:
    """
    Decompose end-to-end inference latency into component stages.
    
    Args:
        stage_latencies: dict mapping stage name -> np.ndarray of latency measurements (ms)
        percentiles: list of percentile values to compute (e.g., [50, 95, 99])
    
    Returns:
        Dictionary with keys: 'e2e_mean', 'e2e_percentiles', 'stage_stats',
                               'bottleneck', 'stage_pct'
    """
    
    column_wise_sum = np.sum(list(stage_latencies.values()), axis=0)
    length = len(list(stage_latencies.values())[0])

    e2e_avg = np.round(column_wise_sum.mean(), 2)

    sorted_vals = np.sort(column_wise_sum)

    e2e_percentiles = {}
    for percent in percentiles:
        k = (percent / 100) * (length - 1)

        # if np.round(k, 0) != k:
        #     k =  
        e2e_percentiles[percent] = np.round(sorted_vals[int(k)], 2)

    stage_stats = {}
    bottleneck_latency = 0
    bottleneck = ""
    stage_pct = {}

    for stage, latencies in stage_latencies.items():
        temp_e2e_percentiles = {}
        sorted_latencies = np.sort(latencies)
        for percent in percentiles:
            k = (percent / 100) * (len(latencies) - 1)
            temp_e2e_percentiles[percent] = np.round(sorted_latencies[int(k)], 2)

        latency_mean = latencies.mean()
        if latency_mean > bottleneck_latency:
            bottleneck_latency = latencies.mean()
            bottleneck = stage

        stage_mean = np.round(latencies.mean(), 2)
        stage_stats[stage] = {"mean": stage_mean,
                            "std": np.round(np.std(latencies), 2),
                            "percentiles": temp_e2e_percentiles}

        stage_pct[stage] = np.round((stage_mean/e2e_avg) * 100, 2)

    return {"e2e_mean": e2e_avg,
            "e2e_percentiles": e2e_percentiles,
            "stage_stats": stage_stats,
            "bottleneck": bottleneck,
            "stage_pct": stage_pct}


        

