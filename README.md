# Aviation incident probability forecast

## 1. Set-up

### 1.1. mdbtools

The system package mdbtools is required to use the Python `access_parser` library. If running is a devcontainer (e.g. in a GitHub Codespace), mdbtools will be installed automatically. If not, install mdbtools via the system package manager. For Debian Linux based systems:

```bash
sudo apt update
sudo apt upgrade
sudo apt install mdbtools
```

### 1.2. Python requirements.txt

If running is a devcontainer (e.g. in a GitHub Codespace), Python requirements will be installed automatically. If not, install via pip with:

```bash
pip install -r requirements.txt
```