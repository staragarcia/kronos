import argparse
import os
import time
import pandas as pd
import numpy as np


def generate_from_template(template_path, out_path, rows, reduction, seed):
    df = pd.read_csv(template_path)
    if df.empty:
        raise ValueError("Template CSV is empty")
    sampled = df.sample(n=rows, replace=True, random_state=seed).reset_index(drop=True)
    
    # Use same realistic churn logic as synthetic
    seasonal_months = np.array([1, 2, 6, 7, 8])
    is_seasonal = np.isin(sampled['join_month'].values, seasonal_months)
    
    base_prob = np.zeros(len(sampled))
    base_prob[is_seasonal] = 0.25  # Seasonal: 25% base
    base_prob[~is_seasonal] = 0.20  # Normal: 20% base
    
    # Adjust for other factors
    base_prob -= sampled['visits_per_week'].values * 0.02  # High visits reduce churn
    base_prob -= (sampled['months_since_joined'].values / 100) * 0.10  # Longer tenure reduces churn
    base_prob += (60 - sampled['age'].values) / 1000  # Age has tiny impact
    base_prob += sampled['payment_delay_days'].values * 0.002  # Payment delays minimal impact
    
    base_prob = np.clip(base_prob, 0.05, 0.50)
    sampled['churned'] = (np.random.RandomState(seed).rand(len(sampled)) < base_prob).astype(int)
    sampled.to_csv(out_path, index=False)
    return sampled


def generate_synthetic(out_path, rows, reduction, seed):
    rs = np.random.RandomState(seed)
    ages = rs.randint(18, 70, size=rows)
    months_since_joined = np.round(np.abs(rs.normal(20, 10, size=rows)), 1)
    join_month = rs.randint(1, 13, size=rows)
    visits_per_week = np.round(np.abs(rs.normal(2.2, 0.8, size=rows)), 1)
    visits_per_week = np.clip(visits_per_week, 0.1, 5.0)
    # Lower payment delays (realistic: most members pay on time)
    payment_delay_days = rs.randint(0, 8, size=rows)
    # 5% chance of higher delays (financial hardship)
    high_delay_mask = rs.random(rows) < 0.05
    payment_delay_days[high_delay_mask] = rs.randint(8, 15, size=high_delay_mask.sum())
    
    # Base probability - much lower and more realistic
    # Seasonal joiners (Jan, Feb, Jun, Jul, Aug) have slightly higher base churn
    seasonal_months = np.array([1, 2, 6, 7, 8])
    is_seasonal = np.isin(join_month, seasonal_months)
    
    base_prob = np.zeros(rows)
    base_prob[is_seasonal] = 0.25  # Seasonal: 25% base
    base_prob[~is_seasonal] = 0.20  # Normal: 20% base
    
    # Adjust for other factors (weakly to avoid inflating)
    base_prob -= visits_per_week * 0.02  # High visits reduce churn
    base_prob -= (months_since_joined / 100) * 0.10  # Longer tenure reduces churn
    base_prob += (60 - ages) / 1000  # Age has tiny impact
    base_prob += payment_delay_days * 0.002  # Payment delays minimal impact
    
    base_prob = np.clip(base_prob, 0.05, 0.50)
    churned = (rs.rand(rows) < base_prob).astype(int)
    out = pd.DataFrame({
        'age': ages,
        'months_since_joined': months_since_joined,
        'join_month': join_month,
        'visits_per_week': visits_per_week,
        'payment_delay_days': payment_delay_days,
        'churned': churned,
    })
    out.to_csv(out_path, index=False)
    return out


def backup_file(path):
    if os.path.exists(path):
        ts = time.strftime('%Y%m%d_%H%M%S')
        new = f"{path}.bak.{ts}"
        os.replace(path, new)
        return new
    return None


def summary(df):
    overall = df['churned'].mean()
    seasonal_months = [1, 2, 6, 7, 8]
    seasonal_mask = df['join_month'].isin(seasonal_months)
    seasonal_churn = df[seasonal_mask]['churned'].mean() if seasonal_mask.any() else float('nan')
    normal_churn = df[~seasonal_mask]['churned'].mean() if (~seasonal_mask).any() else float('nan')
    return overall, seasonal_churn, normal_churn, len(df)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--rows', type=int, default=1000)
    p.add_argument('--output', type=str, default='training_data.csv')
    p.add_argument('--template', type=str, default='training_data.csv')
    p.add_argument('--reduction', type=float, default=0.3,
                   help='Multiply churn probability for ages 25-30 by this factor (0-1)')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--force', action='store_true')
    args = p.parse_args()

    out_path = args.output
    template = args.template

    if os.path.exists(out_path) and not args.force:
        backup = backup_file(out_path)
        if backup:
            print(f"Backed up existing {out_path} to {backup}")

    if os.path.exists(template) and os.path.getsize(template) > 0:
        print(f"Using template {template} to build {out_path} with {args.rows} rows")
        df = generate_from_template(template, out_path, args.rows, args.reduction, args.seed)
    else:
        print(f"No template found. Generating synthetic data to {out_path}")
        df = generate_synthetic(out_path, args.rows, args.reduction, args.seed)

    overall, seasonal_churn, normal_churn, n = summary(pd.read_csv(out_path))
    print(f"Wrote {out_path} ({n} rows). Overall churn rate: {overall:.3f}")
    print(f"  Seasonal joiners (Jan/Feb/Jun/Jul/Aug): {seasonal_churn:.3f}")
    print(f"  Normal joiners: {normal_churn:.3f}")
