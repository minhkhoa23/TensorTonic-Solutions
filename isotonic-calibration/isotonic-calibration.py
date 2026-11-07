from bisect import bisect_right

def calibrate_isotonic(cal_labels: list, cal_probs: list, new_probs: list) -> list:
    """
    Returns a list of calibrated probabilities.
    """
    # Write code here
    pairs = sorted(zip(cal_probs, cal_labels))

    xs = [p for p, y in pairs]

    # Step 2: Pool Adjacent Violators Algorithm
    blocks = []

    for p, y in pairs:
        blocks.append([float(y), 1])

        # Merge blocks if monotonicity is violated
        while len(blocks) >= 2:
            prev = blocks[-2]
            curr = blocks[-1]

            prev_mean = prev[0] / prev[1]
            curr_mean = curr[0] / curr[1]

            if prev_mean <= curr_mean:
                break

            blocks.pop()
            prev[0] += curr[0]
            prev[1] += curr[1]

    # Step 3: Expand fitted values
    fitted = []

    for total, count in blocks:
        mean = total / count
        fitted.extend([mean] * count)

    # Step 4: Calibrate new probabilities
    result = []

    for p in new_probs:

        # Clamp outside calibration range
        if p <= xs[0]:
            result.append(fitted[0])
            continue

        if p >= xs[-1]:
            result.append(fitted[-1])
            continue

        # Find neighboring probabilities
        right = bisect_right(xs, p)
        left = right - 1

        x1, x2 = xs[left], xs[right]
        y1, y2 = fitted[left], fitted[right]

        # Linear interpolation
        t = (p - x1) / (x2 - x1)
        calibrated = y1 + t * (y2 - y1)

        result.append(calibrated)

    return result
