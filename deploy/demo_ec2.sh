#!/usr/bin/env bash
# Host the OceanEmbed Streamlit demo on a fresh Ubuntu 24.04 EC2 box:
#   sudo bash demo_ec2.sh                            -> http(s)://<public-ip>
#   sudo DOMAIN=demo.example.com bash demo_ec2.sh    -> also https://demo.example.com
# nginx serves HTTPS for whichever Let's Encrypt certificates exist under
# /etc/letsencrypt/live/, and plain HTTP until one does; app/README.md has the one-time
# certbot commands for both a bare IP and a domain.
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
apt-get install -y -qq nginx
mkdir -p /var/www/html
IP="$(curl -s --max-time 5 ifconfig.me || echo '')"

# The proxy body lives in its own file and every TLS server block includes it, so the
# websocket settings are written once instead of copied per certificate.
#
# Streamlit talks to the browser over a websocket; everything after the first paint rides
# on it. proxy_http_version/Upgrade/Connection are the three lines that matter -- without
# them the page renders once and then ignores every click, which looks like a broken app
# rather than a broken proxy. proxy_read_timeout matters as much: nginx closes an idle
# connection after 60 s by default, so a demo parked on one slide comes back to
# "Connection lost".
cat > /etc/nginx/oceanembed-proxy.conf <<'PROXY'
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

# One TLS server block per certificate actually held. Certificates are obtained out of
# band (see the HTTPS notes in app/README.md) because issuance needs an email address and
# a Terms of Service acceptance, which do not belong in an unattended script.
tls_block () {          # $1 = server_name, $2 = /etc/letsencrypt/live dir, $3 = extra
  cat <<TLS
server {
    listen 443 ssl $3;
    server_name $1;
    ssl_certificate $2/fullchain.pem;
    ssl_certificate_key $2/privkey.pem;
    include /etc/letsencrypt/options-ssl-nginx.conf;
    ssl_dhparam /etc/letsencrypt/ssl-dhparams.pem;
    include /etc/nginx/oceanembed-proxy.conf;
}
TLS
}

CONF=/etc/nginx/sites-available/oceanembed
: > "$CONF"
HAVE_TLS=""
[ -n "$IP" ] && [ -d "/etc/letsencrypt/live/$IP" ] && HAVE_TLS=1
[ -n "$DOMAIN" ] && [ -d "/etc/letsencrypt/live/$DOMAIN" ] && HAVE_TLS=1

# Port 80 always serves the ACME challenge unredirected -- redirecting it breaks the
# renewal that keeps the site up -- and otherwise redirects to HTTPS once a certificate
# exists, or proxies directly while one does not.
{
  echo 'server {'
  echo '    listen 80 default_server;'
  echo '    server_name _;'
  echo '    location /.well-known/acme-challenge/ { root /var/www/html; }'
  if [ -n "$HAVE_TLS" ]; then
    echo '    location / { return 301 https://$host$request_uri; }'
  else
    echo '    include /etc/nginx/oceanembed-proxy.conf;'
  fi
  echo '}'
} >> "$CONF"

# The IP block is the default_server: it answers anything that arrives without a matching
# Host, including someone typing the raw address.
if [ -n "$IP" ] && [ -d "/etc/letsencrypt/live/$IP" ]; then
  tls_block '_' "/etc/letsencrypt/live/$IP" 'default_server' >> "$CONF"
  READY="https://$IP"
fi
if [ -n "$DOMAIN" ] && [ -d "/etc/letsencrypt/live/$DOMAIN" ]; then
  tls_block "$DOMAIN" "/etc/letsencrypt/live/$DOMAIN" '' >> "$CONF"
  READY="https://$DOMAIN"
fi
[ -n "$HAVE_TLS" ] || READY="http://${IP:-<public-ip>}"

ln -sf "$CONF" /etc/nginx/sites-enabled/oceanembed
rm -f /etc/nginx/sites-enabled/default        # its own default_server would win on :80
nginx -t
systemctl restart nginx
echo "ready: $READY"
