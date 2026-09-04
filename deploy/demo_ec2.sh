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

# The app sits at ~1.1 GB once a viewer has browsed every quarter. That fits 4 GB alone,
# but a second Python process -- app/loader.py's self-check, say -- pushes past it, and
# with no swap the OOM killer gets to choose a victim. It should never get to choose the
# demo. 2 GB of disk is cheaper than a larger instance.
if ! swapon --show | grep -q .; then
  fallocate -l 2G /swapfile
  chmod 600 /swapfile
  mkswap -q /swapfile
  swapon /swapfile
  grep -q '^/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' >> /etc/fstab
fi

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

# A proxy always fronts the app, so Streamlit itself never listens on a public
# interface: port 8501 is reachable only from the box.
BIND=127.0.0.1

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
  # Caddy, purely for the automatic Let's Encrypt certificate. A bare IP cannot have one,
  # so this path needs a real domain; without a domain we serve plain HTTP over nginx.
  apt-get install -y -qq debian-keyring debian-archive-keyring apt-transport-https curl
  curl -1sLf https://dl.cloudsmith.io/public/caddy/stable/gpg.key     | gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
  curl -1sLf https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt     > /etc/apt/sources.list.d/caddy-stable.list
  apt-get update -qq && apt-get install -y -qq caddy
  printf '%s {
	reverse_proxy localhost:8501
}
' "$DOMAIN" > /etc/caddy/Caddyfile
  systemctl restart caddy
  echo "ready: https://$DOMAIN"
else
  apt-get install -y -qq nginx
  IP="$(curl -s --max-time 5 ifconfig.me || echo '')"
  CERTDIR="/etc/letsencrypt/live/$IP"

  # Streamlit talks to the browser over a websocket; everything after the first paint
  # rides on it. proxy_http_version/Upgrade/Connection are the three lines that matter --
  # without them the page renders once and then ignores every click, which looks like a
  # broken app rather than a broken proxy. proxy_read_timeout matters as much: nginx
  # closes an idle connection after 60 s by default, so a demo parked on one slide comes
  # back to "Connection lost".
  cat > /tmp/oceanembed.proxy <<'PROXY'
    location / {
        proxy_pass http://127.0.0.1:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 3600s;
        proxy_send_timeout 3600s;
    }
PROXY

  if [ -d "$CERTDIR" ]; then
    # HTTPS on a bare IP. Let's Encrypt has issued IP certificates since Jan 2026, but
    # only under the 6-day "shortlived" profile and only via a non-nginx authenticator --
    # see the HTTPS notes in app/README.md for the one-time certbot command. The ACME
    # challenge path stays on plain HTTP and is NOT redirected, or renewal breaks itself.
    { echo 'server {'
      echo '    listen 80 default_server;'
      echo '    server_name _;'
      echo '    location /.well-known/acme-challenge/ { root /var/www/html; }'
      echo '    location / { return 301 https://$host$request_uri; }'
      echo '}'
      echo 'server {'
      echo '    listen 443 ssl default_server;'
      echo '    server_name _;'
      echo "    ssl_certificate $CERTDIR/fullchain.pem;"
      echo "    ssl_certificate_key $CERTDIR/privkey.pem;"
      echo '    include /etc/letsencrypt/options-ssl-nginx.conf;'
      echo '    ssl_dhparam /etc/letsencrypt/ssl-dhparams.pem;'
      cat /tmp/oceanembed.proxy
      echo '}'
    } > /etc/nginx/sites-available/oceanembed
    READY="https://$IP"
  else
    { echo 'server {'
      echo '    listen 80 default_server;'
      echo '    server_name _;'
      echo '    location /.well-known/acme-challenge/ { root /var/www/html; }'
      cat /tmp/oceanembed.proxy
      echo '}'
    } > /etc/nginx/sites-available/oceanembed
    READY="http://${IP:-<public-ip>}"
  fi

  mkdir -p /var/www/html
  rm -f /tmp/oceanembed.proxy
  ln -sf /etc/nginx/sites-available/oceanembed /etc/nginx/sites-enabled/oceanembed
  rm -f /etc/nginx/sites-enabled/default        # its own default_server would win on :80
  nginx -t
  systemctl restart nginx
  echo "ready: $READY"
fi
