"""
GitHub Copilot authentication token discovery module.

This module locates GitHub Copilot authentication tokens from various sources:
1. Environment variable GITHUB_TOKEN (explicit override)
2. Local Copilot client files created by IDEs/CLIs:
   - ~/.config/github-copilot/hosts.json
   - ~/.config/github-copilot/apps.json
   - $XDG_CONFIG_HOME/github-copilot/hosts.json
   - $XDG_CONFIG_HOME/github-copilot/apps.json

These JSON files are created by VSCode, JetBrains, gh CLI, or Neovim plugins
when users authenticate with GitHub Copilot.
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional


def find_copilot_token() -> Optional[str]:
    """
    Find GitHub Copilot authentication token from standard locations.
    
    Returns:
        The authentication token if found, None otherwise.
    """
    # 1. Check environment variable (explicit override)
    if token := os.environ.get("GITHUB_TOKEN"):
        logging.info("Found GitHub Copilot token from GITHUB_TOKEN environment variable")
        return token
    
    # 2. Check standard file locations
    candidate_paths = _get_candidate_paths()
    
    for path in candidate_paths:
        if path.exists():
            try:
                token = _extract_token_from_file(path)
                if token:
                    logging.info(f"Found GitHub Copilot token from {path}")
                    return token
            except Exception as e:
                logging.debug(f"Failed to read token from {path}: {e}")
                continue
    
    logging.warning("No GitHub Copilot token found. Please authenticate with an IDE or set GITHUB_TOKEN environment variable.")
    return None


def _get_candidate_paths() -> list[Path]:
    """Get list of candidate paths where Copilot token files might exist."""
    paths = []
    
    # Check XDG_CONFIG_HOME first (if set)
    if xdg_config := os.environ.get("XDG_CONFIG_HOME"):
        xdg_path = Path(xdg_config)
        paths.extend([
            xdg_path / "github-copilot" / "hosts.json",
            xdg_path / "github-copilot" / "apps.json",
        ])
    
    # Check standard ~/.config location
    home = Path.home()
    paths.extend([
        home / ".config" / "github-copilot" / "hosts.json",
        home / ".config" / "github-copilot" / "apps.json",
    ])
    
    return paths


def _extract_token_from_file(path: Path) -> Optional[str]:
    """
    Extract authentication token from a Copilot JSON file.
    
    Typical file format:
    {
        "github.com": {
            "user": "username",
            "oauth_token": "gho_..."
        }
    }
    
    Args:
        path: Path to the JSON file
        
    Returns:
        The token if found, None otherwise.
    """
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Look for github.com entry with token
    if "github.com" in data and isinstance(data["github.com"], dict):
        github_data = data["github.com"]
        
        # Try common token key names
        for key in ["oauth_token", "token", "access_token", "value"]:
            if key in github_data and isinstance(github_data[key], str):
                token = github_data[key]
                if token and token.strip():
                    return token.strip()
    
    # Fallback: scan all nested objects for any string that looks like a token
    for value in data.values():
        if isinstance(value, dict):
            for key in ["oauth_token", "token", "access_token", "value"]:
                if key in value and isinstance(value[key], str):
                    token = value[key]
                    if token and token.strip():
                        return token.strip()
    
    return None
