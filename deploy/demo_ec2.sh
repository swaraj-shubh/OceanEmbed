#!/usr/bin/env bash
# Host the OceanEmbed Streamlit demo on a fresh Ubuntu 24.04 EC2 box. Run once:
#   sudo bash demo_ec2.sh                 -> http://<public-ip>:8501
#   sudo DOMAIN=demo.example.com bash demo_ec2.sh   -> https://demo.example.com
# Idempotent: re-run to redeploy the latest main.
set -euo pipefail

REPO="${REPO:-https://github.com/swaraj-shubh/OceanEmbed.git}"
DOMAIN="${DOMAIN:-}"
APP_USER="${APP_USER:-ubuntu}"
DIR="/home/$APP_USER/OceanEmbed"

apt-get update -qq
apt-get install -y -qq git python3-venv

# Shallow clone: the demo needs the tip of main, not 60 MB of history.
if [ -d "$DIR/.git" ]; then
  sudo -u "$APP_USER" git -C "$DIR" fetch --depth 1 origin main
  sudo -u "$APP_USER" git -C "$DIR" reset --hard origin/main
else
  sudo -u "$APP_USER" git clone --depth 1 "$REPO" "$DIR"
fi

# app/requirements.txt, NOT the root one -- the root pulls torch/cartopy and is not
# needed: predictions are precomputed into app/demo_data/.
sudo -u "$APP_USER" python3 -m venv "$DIR/.venv"
sudo -u "$APP_USER" "$DIR/.venv/bin/pip" install -q --upgrade pip
sudo -u "$APP_USER" "$DIR/.venv/bin/pip" install -q -r "$DIR/app/requirements.txt"

# Bind to localhost only when Caddy fronts it, so 8501 is not separately reachable.
BIND=$([ -n "$DOMAIN" ] && echo 127.0.0.1 || echo 0.0.0.0)

cat > /etc/systemd/system/oceanembed.service <<UNIT
[Unit]
Description=OceanEmbed Streamlit demo
After=network-online.target

[Service]
User=$APP_USER
WorkingDirectory=$DIR
ExecStart=$DIR/.venv/bin/streamlit run app/streamlit_app.py \
  --server.address $BIND --server.port 8501 --server.headless true
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
UNIT

systemctl daemon-reload
systemctl enable --now oceanembed
systemctl restart oceanembed

if [ -n "$DOMAIN" ]; then
  # Caddy for automatic Let's Encrypt TLS. It proxies websockets untouched, which
  # Streamlit needs -- an nginx config that forgets Upgrade/Connection headers gives a
  # page that loads and then never updates.
  apt-get install -y -qq debian-keyring debian-archive-keyring apt-transport-https curl
  curl -1sLf https://dl.cloudsmith.io/public/caddy/stable/gpg.key \
    | gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
  curl -1sLf https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt \
    > /etc/apt/sources.list.d/caddy-stable.list
  apt-get update -qq && apt-get install -y -qq caddy
  printf '%s {\n\treverse_proxy localhost:8501\n}\n' "$DOMAIN" > /etc/caddy/Caddyfile
  systemctl restart caddy
  echo "ready: https://$DOMAIN"
else
  echo "ready: http://$(curl -s --max-time 5 ifconfig.me || echo '<public-ip>'):8501"
fi
