"""Tests for the finance-news CLI environment setup."""

import os
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    ("ssl_cert_file", "fallback_file"),
    [
        (None, None),
        ("/custom/ca.pem", None),
        (None, "ca-certificates.crt"),
    ],
)
def test_cli_maps_only_a_set_ssl_cert_file_to_requests_bundle(tmp_path, ssl_cert_file, fallback_file):
    cli = Path(__file__).parents[1] / "scripts" / "finance-news"
    ca_dir = tmp_path / "system-certs"
    ca_dir.mkdir()
    if fallback_file:
        (ca_dir / fallback_file).touch()

    cli_copy = tmp_path / "finance-news"
    cli_source = cli.read_text()
    for system_path, test_path in (
        ("/etc/ssl/certs/ca-bundle.crt", '"$FINANCE_NEWS_TEST_CA_DIR/ca-bundle.crt"'),
        (
            "/etc/ssl/certs/ca-certificates.crt",
            '"$FINANCE_NEWS_TEST_CA_DIR/ca-certificates.crt"',
        ),
    ):
        assert system_path in cli_source
        cli_source = cli_source.replace(system_path, test_path)
    cli_copy.write_text(cli_source)
    cli_copy.chmod(0o755)

    capture_file = tmp_path / "environment.txt"
    fake_python = tmp_path / "fake-python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        'printf "%s\\n%s\\n" "${SSL_CERT_FILE-<unset>}" '
        '"${REQUESTS_CA_BUNDLE-<unset>}" > "$CA_ENV_CAPTURE"\n'
    )
    fake_python.chmod(0o755)

    env = os.environ.copy()
    env.pop("SSL_CERT_FILE", None)
    env.pop("REQUESTS_CA_BUNDLE", None)
    env.update(
        {
            "CA_ENV_CAPTURE": str(capture_file),
            "FINANCE_NEWS_TEST_CA_DIR": str(ca_dir),
            "PYTHON_BIN": str(fake_python),
        }
    )
    if ssl_cert_file is not None:
        env["SSL_CERT_FILE"] = ssl_cert_file

    subprocess.run(
        ["bash", str(cli_copy), "market"],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    ssl_value, requests_value = capture_file.read_text().splitlines()
    expected_ssl = str(ca_dir / fallback_file) if fallback_file else ssl_cert_file or "<unset>"
    assert ssl_value == expected_ssl
    assert requests_value == expected_ssl
