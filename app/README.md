# OceanEmbed demo

```bash
pip install -r app/requirements.txt
python scripts/build_demo_bundle.py     # once, needs the full repo artifacts
python app/loader.py                    # self-check the bundle
streamlit run app/streamlit_app.py
```

Runs **fully offline**. No torch, no GPU, no network — predictions for the demo window are
precomputed into `app/demo_data/` (~70 MB, committed), so a click is an array lookup.

## What the judge does

1. **① Surface inputs** — the seven satellite fields that are the model's only input.
2. **② Reconstruction** — temperature at any of 15 depths, with a GLORYS side-by-side and a
   difference view.
3. **③ Profile** — click the map, get the 0–1000 m column with the nearest *independent*
   Argo float overlaid and the local RMSE / bias / correlation.
4. **④ Skill** — accuracy by depth against ~6,000 held-out Argo casts, versus the
   climatology floor and the GLORYS ceiling.

## The 90-second path

Pick **5 Dec 2023** (Cyclone Michaung, Bay of Bengal) → tab ② at **100 m** → switch to
*Difference* to show where we depart from the reanalysis → tab ③, click into the Bay →
profile tracks the Argo float → tab ④ for the depth curve. Rehearse it.

## Notes

- The window is **2023-10-01 → 2023-12-31**, inside the test split the model never trained
  on. It contains Cyclone Tej (Arabian Sea) and Cyclone Michaung (Bay of Bengal).
- Bundle values are int16-packed; round-trip error is ~0.0002 °C, versus the model's
  0.786 °C RMSE.
- On-screen profile metrics use `src/argo_eval.interp_profile`, the same acceptance rule as
  every reported number — not a second implementation.
- **Streamlit Cloud:** point the app at `app/streamlit_app.py` and set the dependency file
  to `app/requirements.txt` in Advanced settings, or the root `requirements.txt` (torch,
  cartopy, copernicusmarine) will be installed and the build will time out.
