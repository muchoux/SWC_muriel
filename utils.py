import pandas as pd

def get_metric_by_kwp_interval(df_multi, location, metric, kwp_value):
    """
    Returns the value of electricity corresponding to the kwp value
    - df_multi : DataFrame read with header=[0,1]
    - location : ex "LIMA"
    - metric   : ex "HIT"
    - kwp_value : kwp value wanted
    """
    # Find corresponding columns for metric and location
    candidates = []
    for col in df_multi.columns:
        loc = str(col[1]).strip().upper()
        met = str(df_multi.iloc[0][col]).strip().upper()
        if loc == str(location).strip().upper() and met == str(metric).strip().upper():
            candidates.append(col)
    if not candidates:
        return {"error": f"No data found for location='{location}' and metric='{metric}'."}
    col = candidates[0]

    # kwp columns
    kwp_series = pd.to_numeric(df_multi.iloc[:,0], errors='coerce').dropna().reset_index(drop=True)
    metric_series = pd.to_numeric(df_multi[col], errors='coerce').dropna().reset_index(drop=True)

    # Find the line correponding the interval
    for i in range(len(kwp_series)):
        if kwp_value <= kwp_series.iloc[0]:
            idx = 0
            break
        elif i < len(kwp_series)-1 and kwp_series.iloc[i] < kwp_value <= kwp_series.iloc[i+1]:
            idx = i+1
            break
        elif kwp_value > kwp_series.iloc[-1]:
            idx = len(kwp_series)-1
            break
    else:
        idx = 0  # fallback

    return {
        "location": location,
        "metric": metric,
        "kwp_requested": kwp_value,
        "kwp_selected": kwp_series.iloc[idx],
        "value": metric_series.iloc[idx]
    }