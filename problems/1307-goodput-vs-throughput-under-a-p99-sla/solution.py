def goodput_metrics(requests, sla_ttft_ms, sla_itl_ms, wall_time_s):
    """
    Throughput vs goodput for a decode trace under a TTFT/ITL SLA.

    Returns a dict with throughput_tps, goodput_tps, n_ok, n_total.
    """
    
    throughput_tps = round(sum([r["output_tokens"] for r in requests]) / wall_time_s, 4)


    valid_sla = []
    for r in requests:


        if r["ttft_ms"] <= sla_ttft_ms:
            if len(r["itl_ms"]) == 0 or all([a <= sla_itl_ms for a in r["itl_ms"]]):
                valid_sla.append(r["output_tokens"])


    goodput_tps = round(sum(valid_sla) / wall_time_s, 4)

    n_ok = len(valid_sla)

    n_total = len(requests)

    return {"throughput_tps": throughput_tps,
            "goodput_tps": goodput_tps,
            "n_ok": n_ok,
            "n_total": n_total}