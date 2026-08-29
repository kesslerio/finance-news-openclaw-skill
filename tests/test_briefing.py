import sys
from pathlib import Path
import json
import pytest
from unittest.mock import Mock, patch
import subprocess

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from briefing import generate_and_send

def test_generate_and_send_success():
    # Mock subprocess.run for summarize.py
    mock_briefing_data = {
        "macro_message": "Macro Summary",
        "portfolio_message": "Portfolio Summary",
        "summary": "Full Summary"
    }
    
    with patch("briefing.subprocess.run") as mock_run:
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps(mock_briefing_data)
        mock_run.return_value = mock_result
        
        args = Mock()
        args.time = "morning"
        args.style = "briefing"
        args.lang = "en"
        args.deadline = 300
        args.fast = False
        args.llm = False
        args.model = "ornith"
        args.debug = False
        args.json = True
        args.send = False
        
        result = generate_and_send(args)
        
        assert result == "Macro Summary"
        assert mock_run.called
        # Check if summarize.py was called with correct args
        call_args = mock_run.call_args[0][0]
        assert "summarize.py" in str(call_args[1])
        assert "--time" in call_args
        assert "morning" in call_args
        assert call_args[call_args.index("--model") + 1] == "ornith"


def test_generate_and_send_honors_explicit_manual_ds4_for_analysis():
    mock_briefing_data = {"macro_message": "Manual analysis"}

    with patch("briefing.subprocess.run") as mock_run:
        mock_run.return_value = Mock(returncode=0, stdout=json.dumps(mock_briefing_data))
        args = Mock(
            time="morning",
            style="analysis",
            lang="en",
            deadline=300,
            fast=False,
            llm=False,
            model="ds4",
            debug=False,
            json=True,
            send=False,
        )

        generate_and_send(args)

        call_args = mock_run.call_args[0][0]
        assert call_args[call_args.index("--model") + 1] == "ds4"

def test_generate_and_send_with_whatsapp():
    mock_briefing_data = {
        "macro_message": "Macro Summary",
        "portfolio_message": "Portfolio Summary"
    }
    
    with patch("briefing.subprocess.run") as mock_run, \
         patch("briefing.send_to_whatsapp") as mock_send:
        
        # First call is summarize.py
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps(mock_briefing_data)
        mock_run.return_value = mock_result
        
        args = Mock()
        args.time = "evening"
        args.style = "briefing"
        args.lang = "en"
        args.deadline = None
        args.fast = True
        args.llm = False
        args.model = "ornith"
        args.json = False
        args.send = True
        args.group = "Test Group"
        args.debug = False
        
        generate_and_send(args)
        
        # Check if send_to_whatsapp was called for both messages
        assert mock_send.call_count == 2
        mock_send.assert_any_call("Macro Summary", "Test Group")
        mock_send.assert_any_call("Portfolio Summary", "Test Group")

def test_generate_and_send_failure():
    with patch("briefing.subprocess.run") as mock_run:
        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stderr = "Error occurred"
        mock_run.return_value = mock_result
        
        args = Mock()
        args.time = "morning"
        args.style = "briefing"
        args.lang = "en"
        args.deadline = None
        args.fast = False
        args.llm = False
        args.model = "ornith"
        args.json = False
        args.send = False
        args.debug = False
        
        with pytest.raises(SystemExit):
            generate_and_send(args)
