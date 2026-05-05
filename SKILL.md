---
name: finance-news
description: "Market news briefings with AI summaries and price alerts. Aggregates headlines from US/Europe/Japan markets. Use when: 'stock news', 'market updates', 'morning briefing', 'evening market wrap', 'financial headlines', 'price alerts', 'what happened in the market'. Supports WhatsApp delivery and English/German output. NOT for fundamental analysis or scoring (use equity-research). NOT for raw financial data queries (use openbb)."
---

# Finance News Skill

AI-powered market news briefings with configurable language output and automated delivery via WhatsApp/Telegram.

## First-Time Setup

Run the interactive setup wizard:

```bash
finance-news setup
```

The wizard configures RSS feeds, markets (US/Europe/Japan/Asia), delivery channels (WhatsApp/Telegram), language (English/German), and cron schedule.

Configure individual sections:

```bash
finance-news setup --section feeds     # Just RSS feeds
finance-news setup --section delivery  # Just delivery channels
finance-news setup --section schedule  # Just cron schedule
finance-news setup --reset             # Reset to defaults
```

Verify setup completed correctly:

```bash
finance-news config                    # Show current config
finance-news briefing --morning        # Test a dry-run briefing
```

## Quick Start

```bash
finance-news briefing --morning                                  # Morning briefing
finance-news briefing --evening --send --group "Market Briefing" # Evening + WhatsApp
finance-news market                                              # Market overview
finance-news portfolio                                           # Portfolio news
finance-news news AAPL                                           # Ticker-specific news
```

See [COMMANDS.md](COMMANDS.md) for the full CLI reference including portfolio management, cron setup, and configuration details.

## Cron Jobs

```bash
# Add morning briefing (6:30 AM PT, weekdays)
openclaw cron add --schedule "30 6 * * 1-5" \
  --timezone "America/Los_Angeles" \
  --command "bash ~/clawd/skills/finance-news/cron/morning.sh"

# Add evening briefing (1:00 PM PT, weekdays)
openclaw cron add --schedule "0 13 * * 1-5" \
  --timezone "America/Los_Angeles" \
  --command "bash ~/clawd/skills/finance-news/cron/evening.sh"
```

Verify cron jobs are active:

```bash
openclaw cron list
bash ~/clawd/skills/finance-news/cron/morning.sh  # Manual test run
```

## Integration

Combine with OpenBB for detailed quotes before news:

```bash
openbb-quote AAPL && finance-news news AAPL
```

Run briefings via [Lobster](https://github.com/openclaw/lobster) for approval gates and resumability — see [workflows/README.md](workflows/README.md) for full documentation:

```bash
lobster "workflows.run --file workflows/briefing.yaml"
```

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Gemini not working | Run `gemini` and follow the login flow to authenticate |
| RSS feeds timing out | Check network; WSJ/Barron's may need subscription cookies; CNBC/Yahoo always work |
| WhatsApp delivery failing | Verify group exists and bot has access; run `openclaw doctor` |
