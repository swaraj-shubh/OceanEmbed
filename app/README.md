# OceanEmbed demo

```bash
pip install -r app/requirements.txt
python scripts/build_demo_bundle.py     # once, needs the full repo artifacts
python app/loader.py                    # self-check the bundle
streamlit run app/streamlit_app.py
```

Runs **fully offline**. No torch, no GPU, no network — predictions for the whole test
split are precomputed into `app/demo_data/` (478 MB, committed), so a click is an array
lookup.

A hosted copy runs at **https://65.2.207.204** — see *Deployment* below.

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

- The bundle covers the **full test split**, every day the model can predict:
  **2023-01-07 → 2024-12-31** (725 days), chunked into 8 calendar-quarter files so no single
  tracked file exceeds GitHub's 100 MB limit -- `app/loader.py` loads the quarter the
  selected date falls in and keeps **at most two** resident (`chunk_cache`,
  `max_entries=2`). That bound matters: a quarter costs ~150 MB decoded, so an unbounded
  cache reaches ~3.2 GB across all eight, past Streamlit Cloud's 2.7 GB ceiling. Measured
  on the deployed Linux box, resident memory rises to **~1.1 GB** and then plateaus there
  no matter how many quarters are browsed. The scripted path's date (5 Dec 2023,
  Cyclone Michaung) still works unchanged; the window used to stop at 2023-12-31 and now
  simply keeps going.
- Bundle values are int16-packed; round-trip error is ~0.0002 °C, versus the model's
  0.786 °C RMSE.
- On-screen profile metrics use `src/argo_eval.interp_profile`, the same acceptance rule as
  every reported number — not a second implementation.

## Deployment

**EC2 (what is running now).** `deploy/demo_ec2.sh` on a fresh Ubuntu 24.04 box: shallow
clone, venv from `app/requirements.txt`, a systemd unit (`oceanembed.service`) so the app
survives a crash or a reboot, and optional Caddy for TLS when `DOMAIN` is set. Idempotent —
re-run it to redeploy the tip of `main`.

```bash
scp -i key.pem deploy/demo_ec2.sh ubuntu@<ip>:
ssh -i key.pem ubuntu@<ip> 'sudo bash demo_ec2.sh'          # http://<ip>, https if a cert exists
ssh -i key.pem ubuntu@<ip> 'sudo DOMAIN=demo.example.com bash demo_ec2.sh'   # https
```

Size it at **t3.medium** (4 GB): the ~1.1 GB steady state leaves a t3.small no room at
all. The script also adds a 2 GB swapfile, which is not decoration -- running
`app/loader.py`'s self-check next to the live app briefly needs ~2.3 GB, and on the
swapless box the OOM killer fired four times before that was in place. It picked the
self-check every time and the service survived (`NRestarts=0`), but it was under no
obligation to.

Streamlit binds `127.0.0.1` in both cases and a proxy fronts it, so port 8501 is never
publicly exposed. nginx takes 80/443 for a bare IP; Caddy takes 443 when `DOMAIN` is set.
Open 80 and 443 in the security group.

### HTTPS on a bare IP

Let's Encrypt has issued certificates for IP addresses since January 2026, so the demo has
real TLS with no domain at all. Three constraints make the command non-obvious, and each
one is a separate error message if you miss it:

- The IP goes in `--ip-address`, not `-d`; `-d` is rejected outright.
- IP certificates are only issued under the **`shortlived` profile** — 6-day certificates.
- Certbot's nginx plugin supports neither *installing* nor *authenticating* them, so it
  has to be `certonly --webroot`. That is the better choice anyway: `--standalone` would
  stop nginx on every renewal, which at a 6-day lifetime means every few days, possibly
  mid-demo.

Ubuntu's `apt` certbot (2.9.0) predates IP support; the snap (5.8.0) has it.

```bash
sudo snap install --classic certbot && sudo ln -sf /snap/bin/certbot /usr/bin/certbot
sudo certbot certonly --webroot -w /var/www/html --ip-address <ip>   --required-profile shortlived --email you@example.com --agree-tos --no-eff-email -n
printf '#!/bin/sh
systemctl reload nginx
'   | sudo tee /etc/letsencrypt/renewal-hooks/deploy/reload-nginx.sh
sudo chmod +x /etc/letsencrypt/renewal-hooks/deploy/reload-nginx.sh
sudo certbot renew --dry-run          # must say "no renewal failures"
```

Re-run `demo_ec2.sh` afterwards: it detects `/etc/letsencrypt/live/<ip>/` and writes the
443 server block plus an 80→443 redirect, keeping `/.well-known/acme-challenge/` on plain
HTTP and *unredirected* — redirecting it breaks the renewal that keeps the site up.

**A 6-day certificate makes renewal load-bearing**, far more than the usual 90-day one.
`snap.certbot.renew.timer` runs twice daily and the deploy hook reloads nginx; the dry run
above is the check that this works. `http2 on;` is deliberately absent — that directive
needs nginx ≥1.25 and Ubuntu 24.04 ships 1.24.

The nginx config is three lines of substance and two of them are easy to omit.
`proxy_http_version 1.1` with the `Upgrade`/`Connection` headers is what lets Streamlit's
websocket through -- without it the page paints once, returns a healthy 200, and then
ignores every click, which reads as a broken app rather than a broken proxy. And
`proxy_read_timeout 3600s` overrides nginx's 60 s default, which would otherwise drop the
idle socket while you talk over a slide and leave the judge looking at "Connection lost".
Verify with `curl -i -H 'Connection: Upgrade' -H 'Upgrade: websocket' -H
'Sec-WebSocket-Version: 13' -H 'Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ=='
http://<ip>/_stcore/stream` -- it must answer **101 Switching Protocols**, not 200.

The
address is an **Elastic IP** (`eipalloc-048b0502fbb0f9f5d`), so it survives a stop/start --
an auto-assigned one does not, and every link in these docs would rot the first time the
box is stopped to save money. Release the allocation if the demo is ever torn down: an
Elastic IP keeps billing while it sits unassociated.

**Streamlit Cloud** also fits (~1.1 GB against its 2.7 GB cap). Point it at
`app/streamlit_app.py`; there is no dependency-file setting to change — Community Cloud
searches the entrypoint's directory before the repo root, so `app/requirements.txt` wins
and the root one (torch, cartopy, copernicusmarine, which would time the build out) is
ignored. Its one catch is that apps sleep after 12 h without traffic; a free uptime
monitor pinging the URL keeps it awake for demo day.
