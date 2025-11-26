from observability.obs import get_metrics

def format_observability_dashboard() -> str:
    metrics = get_metrics()

    counters = metrics.get("counters", {})
    latencies = metrics.get("latencies", {})
    timings = metrics.get("timings", {})

    lines = []
    lines.append("📊 **OBSERVABILITY DASHBOARD**")
    lines.append("----------------------------------")

    # Timings
    for key, val in timings.items():
        lines.append(f"**{key.replace('_', ' ').title()}**: {val:.2f}s")

    # Latencies
    for key, val in latencies.items():
        lines.append(f"**{key.replace('_', ' ').title()}**: {val:.2f}s")

    # Counters
    for key, val in counters.items():
        lines.append(f"**{key.replace('_', ' ').title()}**: {val}")

    return "\n".join(lines)
