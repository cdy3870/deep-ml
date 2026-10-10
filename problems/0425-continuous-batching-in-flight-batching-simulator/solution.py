from collections import deque
import numpy as np

def continuous_batching_sim(requests: list[dict], max_batch_size: int) -> dict:
    """
    Simulate continuous batching for LLM inference.
    
    Args:
        requests: List of dicts with 'arrival_time' (int) and 'tokens_needed' (int)
        max_batch_size: Maximum concurrent sequences in the batch
    
    Returns:
        Dict with 'total_time', 'avg_latency', 'avg_ttft', 'throughput'
    """
    current_time = 0
    waiting_queue = []
    current_concurrent = 0
    slots = []

    sorted_requests = {}
    completion_times = []
    total_tokens = 0
    for r in requests:
        if r["arrival_time"] in sorted_requests:
            sorted_requests[r["arrival_time"]].append(r)
        else:
            sorted_requests[r["arrival_time"]] = [r]
        total_tokens += r["tokens_needed"]

    num_completed = 0
    start_times = []
    while num_completed < len(requests):
        if current_time in sorted_requests:
            for request in sorted_requests[current_time]:
                if current_concurrent < max_batch_size:
                    request["start_time"] = current_time
                    start_times.append(request)
                    current_concurrent += 1
                    slots.append(request)
                else:
                    waiting_queue.append(request)

        for request in slots[:]:          # (1) iterate over a copy
            request["tokens_needed"] -= 1
            if request["tokens_needed"] == 0:
                slots.remove(request)
                current_concurrent -= 1
                num_completed += 1
                request["completion_time"] = current_time
                completion_times.append(request)
                # (2) replace your inner for-loop with this
                while waiting_queue and current_concurrent < max_batch_size:
                    nxt = waiting_queue.pop(0)      # (3) actually remove it from the queue
                    nxt["start_time"] = current_time + 1
                    start_times.append(nxt)
                    current_concurrent += 1         # (3) and count it
                    slots.append(nxt)

            
        
        current_time += 1

    avg_latency = np.round(sum([r["completion_time"] - r["arrival_time"] for r in completion_times]) / len(requests), 4)
    throughput = np.round(total_tokens / current_time, 4)
    avg_ttft = np.round(sum([r["start_time"] - r["arrival_time"] for r in start_times]) / len(requests), 4)
    # print(start_times)

    return {"total_time": current_time,
            "avg_latency": avg_latency,
            "avg_ttft": avg_ttft,
            "throughput": throughput}

        


        

