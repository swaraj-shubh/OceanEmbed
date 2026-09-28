# OceanEmbed demo

```bash
pip install -r app/requirements.txt
streamlit run app/streamlit_app.py
```

Runs fully offline. No torch, no GPU, no network: predictions for the whole test split
(2023-01-07 to 2024-12-31) are precomputed into `app/demo_data/`, so a click is an array
lookup. A hosted copy runs at **<https://oter.shubhh.xyz>**.

## What the judge does

1. **Surface inputs** - the seven satellite fields that are the model's only input.
2. **Reconstruction** - temperature at any of 15 depths, with a GLORYS side-by-side and a
   difference view.
3. **Profile** - click the map, get the 0-1000 m column with the nearest independent Argo
   float overlaid and the local RMSE / bias / correlation.
4. **Skill** - accuracy by depth against ~6,000 held-out Argo casts, versus the climatology
   floor and the GLORYS ceiling.

## The 90-second path

Pick 5 Dec 2023 (Cyclone Michaung, Bay of Bengal), tab Reconstruction at 100 m, switch to
Difference to show where we depart from the reanalysis, tab Profile, click into the Bay
(profile tracks the Argo float), tab Skill for the depth curve.

Deployment (EC2 + nginx + HTTPS) is documented in `deploy/demo_ec2.sh`.
