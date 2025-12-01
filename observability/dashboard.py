from observability.obs import get_metrics

def format_observability_dashboard() -> str:
    metrics = get_metrics()

    counters = metrics.get("counters", {})
    latencies = metrics.get("latencies", {})
    timings = metrics.get("timings", {})

    lines = []
    lines.append("📊 **OBSERVABILITY DASHBOARD**")
    lines.append("----------------------------------")

    # Timings - each on its own line with double newline for explicit separation
    for key, val in timings.items():
        metric_name = key.replace('_', ' ').title()
        lines.append(f"**{metric_name}**: {val:.2f}s")

    if timings:
        lines.append("")

    # Latencies - each on its own line with double newline for explicit separation
    for key, val in latencies.items():
        metric_name = key.replace('_', ' ').title()
        lines.append(f"**{metric_name}**: {val:.2f}s")

    if latencies:
        lines.append("")

    # Counters - each on its own line with double newline for explicit separation
    for key, val in counters.items():
        metric_name = key.replace('_', ' ').title()
        lines.append(f"**{metric_name}**: {val}")

    # Use double newlines between each metric to ensure they appear on separate lines
    # This prevents markdown from collapsing them into a single line
    # Join all lines, then add double newline after each metric line
    result_parts = []
    for i, line in enumerate(lines):
        result_parts.append(line)
    
    result = "\n".join(result_parts)
    # Add extra newline at the end
    return result + "\n"
