# Running as a Linux Service (systemd)

This section describes how to run **Imbeddings** as a background service on Linux using **systemd**, and how to view its logs.

Before applying this, run Imbeddings at least once and confirm it works (including tests).

## 1. Create a systemd service file

Create the service definition:

```bash
sudo nano /etc/systemd/system/imbeddings.service
```

Paste the following and adjust paths:

```ini
[Unit]
Description=Imbeddings (FastAPI image-embeddings service)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=appuser # Adjust this
Group=appuser # Adjust this

WorkingDirectory=/path/to/imbeddings  # Adjust this
EnvironmentFile=/path/to/imbeddings/.env  # Adjust this

# Safe defaults
Environment="IMBEDDINGS_HOST=127.0.0.1"
Environment="IMBEDDINGS_PORT=8000"

# Adjust this
ExecStart=/path/to/imbeddings/.venv/bin/uvicorn service.main:app \
  --host ${IMBEDDINGS_HOST} \
  --port ${IMBEDDINGS_PORT}

Restart=always
RestartSec=2

[Install]
WantedBy=multi-user.target
```

**Notes**

* All paths must be **absolute**
* Replace `/path/to/imbeddings` with the actual project directory
* The service runs as a non-root user (`appuser`)

## 2. Reload systemd and start the service

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now imbeddings.service
```

## 3. Check service status

```bash
systemctl status imbeddings.service
```

If the service fails to start:

```bash
journalctl -u imbeddings.service -xe
```

## 4. Viewing Logs

The service logs are collected automatically by **systemd-journald**.

Follow logs in real time:

```bash
journalctl -u imbeddings.service -f
```

View recent logs:

```bash
journalctl -u imbeddings.service -n 100
```

