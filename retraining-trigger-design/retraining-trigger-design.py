def retraining_policy(daily_stats: list, config: dict) -> list:
    """
    Returns a list of retraining day numbers.
    """
    # Write code here
    retraining_days = []

    budget = config["budget"]
    retrain_cost = config["retrain_cost"]

    last_retrain_day = -config["cooldown"]
    staleness = 0

    for stats in daily_stats:
        day = stats["day"]
        staleness += 1

        # 3 điều kiện trigger
        drift_trigger = stats["drift_score"] > config["drift_threshold"]

        performance_trigger = (
            stats["performance"] < config["performance_threshold"]
        )

        staleness_trigger = (
            staleness >= config["max_staleness"]
        )

        trigger = (
            drift_trigger
            or performance_trigger
            or staleness_trigger
        )

        # Điều kiện vận hành
        cooldown_ok = (
            day - last_retrain_day >= config["cooldown"]
        )

        budget_ok = budget >= retrain_cost

        if trigger and cooldown_ok and budget_ok:
            retraining_days.append(day)

            last_retrain_day = day
            staleness = 0
            budget -= retrain_cost

    return retraining_days