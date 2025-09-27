# Production Deployment Guide for Live Trading Bot

## Architecture Overview

```
┌─────────────┐      ┌──────────────┐      ┌─────────────┐
│   Netlify   │──────│  DigitalOcean│──────│  Supabase   │
│  (Frontend) │ HTTPS│   (Backend)  │ SQL  │  (Database) │
└─────────────┘      └──────────────┘      └─────────────┘
                            │
                            │ WebSocket
                            ▼
                     ┌──────────────┐
                     │ Binance.US   │
                     │     API       │
                     └──────────────┘
```

## Phase 1: Backend Deployment (DigitalOcean/AWS EC2)

### 1.1 Create VPS Instance
```bash
# DigitalOcean Droplet Specs (Recommended)
- Size: $20/month (2 GB RAM, 2 vCPUs)
- Region: NYC or SFO (closest to Binance.US servers)
- OS: Ubuntu 22.04 LTS
- Enable backups: Yes
- Enable monitoring: Yes
```

### 1.2 Server Setup Script
```bash
#!/bin/bash
# Run on fresh Ubuntu 22.04 server

# Update system
sudo apt update && sudo apt upgrade -y

# Install Node.js 20
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get install -y nodejs

# Install PM2 for process management
sudo npm install -g pm2

# Install nginx for reverse proxy
sudo apt install -y nginx

# Install certbot for SSL
sudo apt install -y certbot python3-certbot-nginx

# Setup firewall
sudo ufw allow 22
sudo ufw allow 80
sudo ufw allow 443
sudo ufw --force enable

# Create app directory
mkdir -p /var/www/trading-bot
cd /var/www/trading-bot
```

### 1.3 Deploy Backend Code
```bash
# On your local machine
cd backend
npm run build

# Create deployment package
tar -czf backend.tar.gz dist package.json package-lock.json

# Upload to server
scp backend.tar.gz root@your-server-ip:/var/www/trading-bot/

# On server
cd /var/www/trading-bot
tar -xzf backend.tar.gz
npm ci --production
```

### 1.4 Environment Configuration
```bash
# Create production .env file on server
cat > /var/www/trading-bot/.env << EOF
NODE_ENV=production
PORT=3001

# Binance.US Production Keys (KEEP SECURE!)
BINANCE_API_KEY=your_production_api_key
BINANCE_SECRET_KEY=your_production_secret_key
BINANCE_BASE_URL=https://api.binance.us

# Database (Supabase)
DATABASE_URL=postgresql://[user]:[password]@[host]:5432/postgres
SUPABASE_URL=https://ewezuuywerilgnhhzzho.supabase.co
SUPABASE_ANON_KEY=your_anon_key
SUPABASE_SERVICE_KEY=your_service_key

# JWT Secret (Generate new one!)
JWT_SECRET=$(openssl rand -base64 32)

# Redis (Optional but recommended)
REDIS_HOST=localhost
REDIS_PORT=6379

# Monitoring
SENTRY_DSN=your_sentry_dsn
LOG_LEVEL=info
EOF

# Secure the file
chmod 600 .env
```

### 1.5 PM2 Configuration
```javascript
// ecosystem.config.js
module.exports = {
  apps: [{
    name: 'trading-bot',
    script: './dist/server.js',
    instances: 1,
    exec_mode: 'fork',
    autorestart: true,
    watch: false,
    max_memory_restart: '1G',
    env: {
      NODE_ENV: 'production'
    },
    error_file: '/var/log/pm2/trading-bot-error.log',
    out_file: '/var/log/pm2/trading-bot-out.log',
    log_file: '/var/log/pm2/trading-bot-combined.log',
    time: true,
    kill_timeout: 5000,
    listen_timeout: 10000,
    cron_restart: '0 0 * * *', // Daily restart at midnight
  }]
};
```

### 1.6 Nginx Configuration
```nginx
# /etc/nginx/sites-available/trading-bot
server {
    listen 80;
    server_name api.yourdomain.com;

    location / {
        proxy_pass http://localhost:3001;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # WebSocket support
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
}
```

### 1.7 Start Services
```bash
# Start PM2
pm2 start ecosystem.config.js
pm2 save
pm2 startup

# Enable nginx
sudo ln -s /etc/nginx/sites-available/trading-bot /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx

# Get SSL certificate
sudo certbot --nginx -d api.yourdomain.com
```

## Phase 2: Frontend Deployment (Netlify)

### 2.1 Update Frontend Configuration
```javascript
// src/config/api.ts
const API_URL = process.env.NODE_ENV === 'production'
  ? 'https://api.yourdomain.com'
  : 'http://localhost:3001';

export const config = {
  API_URL,
  WS_URL: API_URL.replace('https', 'wss').replace('http', 'ws'),
};
```

### 2.2 Build and Deploy
```bash
# Build production frontend
npm run build

# Deploy to Netlify
# Option 1: CLI
npm install -g netlify-cli
netlify deploy --prod --dir=dist

# Option 2: Git integration
# Push to GitHub and connect repo to Netlify
```

### 2.3 Netlify Environment Variables
```
VITE_API_URL=https://api.yourdomain.com
VITE_SUPABASE_URL=https://ewezuuywerilgnhhzzho.supabase.co
VITE_SUPABASE_ANON_KEY=your_anon_key
```

## Phase 3: Monitoring & Alerts

### 3.1 Health Check Monitoring
```bash
# setup-monitoring.sh
#!/bin/bash

# Install monitoring tools
npm install -g pm2-logrotate
pm2 install pm2-logrotate

# Configure log rotation
pm2 set pm2-logrotate:max_size 100M
pm2 set pm2-logrotate:retain 7

# Setup health check cron
cat > /etc/cron.d/trading-bot-health << EOF
*/5 * * * * root curl -f http://localhost:3001/health || pm2 restart trading-bot
EOF
```

### 3.2 Critical Alerts Setup
```javascript
// monitoring/alerts.js
const axios = require('axios');

// Telegram Alert Bot
const TELEGRAM_BOT_TOKEN = 'your_bot_token';
const TELEGRAM_CHAT_ID = 'your_chat_id';

async function sendAlert(message) {
  await axios.post(`https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage`, {
    chat_id: TELEGRAM_CHAT_ID,
    text: `🚨 TRADING BOT ALERT:\n${message}`,
  });
}

// Monitor critical events
pm2.launchBus((err, bus) => {
  bus.on('process:exception', (data) => {
    sendAlert(`Process crashed: ${data.process.name}`);
  });
});
```

### 3.3 Trading Safeguards
```javascript
// backend/src/config/tradingLimits.ts
export const PRODUCTION_LIMITS = {
  MAX_POSITION_SIZE_USD: 1000,
  MAX_DAILY_LOSS_USD: 500,
  MAX_OPEN_POSITIONS: 5,
  MIN_BALANCE_USD: 100,
  EMERGENCY_STOP_LOSS_PERCENT: 10,

  // Rate limits
  MAX_ORDERS_PER_MINUTE: 10,
  MAX_API_CALLS_PER_SECOND: 5,

  // Circuit breakers
  CONSECUTIVE_LOSSES_BEFORE_PAUSE: 3,
  PAUSE_DURATION_MINUTES: 60,
};
```

## Phase 4: Go Live Checklist

### Pre-Launch
- [ ] Backup database
- [ ] Test emergency stop functionality
- [ ] Verify all API endpoints working
- [ ] Check WebSocket connections stable
- [ ] Confirm rate limiting configured
- [ ] Test with small amounts first
- [ ] Setup monitoring alerts
- [ ] Document recovery procedures

### Launch Day
1. Start with paper trading for 24 hours
2. Switch to live with minimal position sizes
3. Monitor closely for first 48 hours
4. Gradually increase position sizes

### Post-Launch Monitoring
- Check logs every 4 hours: `pm2 logs`
- Monitor system resources: `pm2 monit`
- Track API rate limits
- Review daily P&L
- Check for missed trades

## Emergency Procedures

### Stop All Trading
```bash
# SSH into server
ssh root@your-server-ip

# Stop trading immediately
curl -X POST http://localhost:3001/api/v1/trading/emergency-stop

# Or restart service
pm2 restart trading-bot
```

### Rollback Deployment
```bash
# Keep previous version backup
cp -r /var/www/trading-bot /var/www/trading-bot.backup

# To rollback
pm2 stop trading-bot
mv /var/www/trading-bot /var/www/trading-bot.failed
mv /var/www/trading-bot.backup /var/www/trading-bot
pm2 start trading-bot
```

## Cost Breakdown

### Monthly Costs
- DigitalOcean VPS: $20-40
- Netlify (Frontend): Free
- Supabase (Database): Free tier
- Domain name: $1/month
- SSL Certificate: Free (Let's Encrypt)
- Total: ~$25-45/month

### Optional Add-ons
- Redis Cloud: $5-30/month
- Sentry Monitoring: $26/month
- Datadog APM: $31/month
- Backup storage: $5/month

## Security Best Practices

1. **API Key Security**
   - Never commit keys to Git
   - Use environment variables
   - Rotate keys monthly
   - IP whitelist on Binance.US

2. **Server Security**
   - Enable firewall (ufw)
   - Disable root SSH (use sudo user)
   - Setup fail2ban
   - Regular security updates

3. **Trading Security**
   - Implement position limits
   - Use stop-loss orders
   - Monitor for anomalies
   - Daily reconciliation

## Support & Maintenance

### Daily Tasks
- Check system health
- Review trading logs
- Monitor positions
- Check for errors

### Weekly Tasks
- Review performance metrics
- Update strategies
- Check for updates
- Backup database

### Monthly Tasks
- Rotate API keys
- Security audit
- Performance review
- Cost optimization

---

## Quick Start Commands

```bash
# Check status
pm2 status
pm2 logs trading-bot

# Restart
pm2 restart trading-bot

# Monitor
pm2 monit

# Emergency stop
curl -X POST http://localhost:3001/api/v1/trading/emergency-stop
```

## Troubleshooting

### Backend not responding
```bash
pm2 logs trading-bot --lines 100
pm2 restart trading-bot
```

### WebSocket disconnections
- Check nginx configuration
- Verify firewall rules
- Check Binance API status

### High memory usage
```bash
pm2 restart trading-bot
pm2 set trading-bot --max-memory-restart 1G
```

### API rate limits
- Reduce trading frequency
- Implement request caching
- Use WebSocket for real-time data