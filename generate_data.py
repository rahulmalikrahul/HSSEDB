import pandas as pd
import numpy as np

# Set seed for consistent results
np.random.seed(42)

def generate_dummy_data(output_file='advanced_hsse_data.csv'):
    months = pd.date_range(start='2024-01-01', periods=24, freq='ME').strftime('%Y-%m').tolist()
    sites = ['Site_Alpha', 'Site_Bravo', 'Site_Charlie', 'Site_Delta']
    contractors = ['Global_Build', 'Safety_First_Inc', 'Rapid_Construct', 'Elite_Engineering']
    
    # Base risk multipliers (Site Bravo is dangerous, Site Delta is safe)
    site_risk_map = {'Site_Alpha': 1.0, 'Site_Bravo': 2.5, 'Site_Charlie': 1.2, 'Site_Delta': 0.6}
    # Contractor rating (Lower is better/safer)
    contractor_map = {'Global_Build': 1.1, 'Safety_First_Inc': 0.8, 'Rapid_Construct': 1.5, 'Elite_Engineering': 0.9}

    rows = []

    for month in months:
        for site in sites:
            for contractor in contractors:
                # 1. Generate Exposure (Man Hours)
                man_hours = np.random.randint(5000, 25000)
                
                # 2. Generate Leading Indicators
                # We make these slightly correlated to the Site/Contractor quality
                maint_compliance = np.clip(np.random.normal(85, 10) if contractor_map[contractor] < 1.0 else np.random.normal(70, 15), 0, 100)
                audits = np.random.randint(2, 15)
                observations = np.random.randint(10, 80)

                # 3. The Poisson Lambda (Hazard Rate) Calculation
                # Lambda = (Exposure * BaseRisk * SiteFactor * ContractorFactor) / (SafetyActivities)
                base_lambda = (man_hours / 10000) * 0.5
                site_factor = site_risk_map[site]
                cont_factor = contractor_map[contractor]
                
                # Safety activities "shield" the site
                barrier_strength = (maint_compliance / 100) * (1 + (audits * 0.05) + (observations * 0.02))
                
                final_lambda = (base_lambda * site_factor * cont_factor) / barrier_strength
                
                # 4. Generate the Incident Count
                incidents = np.random.poisson(final_lambda)

                rows.append({
                    'Month': month,
                    'Site': site,
                    'Contractor': contractor,
                    'Man_Hours': man_hours,
                    'Audits': audits,
                    'Observations': observations,
                    'Maint_Compliance': round(maint_compliance, 2),
                    'Incidents': incidents
                })

    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False)
    print(f"✅ Success! Generated {len(df)} rows of data in '{output_file}'")

if __name__ == "__main__":
    generate_dummy_data()