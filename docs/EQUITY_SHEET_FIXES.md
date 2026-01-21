# Equity Sheet Fixes

## NRR Column (Column Q) - Range Values Fix

**Problem:** Values like "115-120%", "125%+", "N/A" in NRR column cause #VALUE! errors in MSS Score formula (columns Y/Z).

**Root cause:** Excel/Sheets formulas cannot perform math operations on text ranges.

**Solution:** Convert all NRR values to single numeric percentages.

### Conversion Rules

| Original | Fixed | Calculation |
|----------|-------|-------------|
| 115-120% | 117.5% | (115+120)/2 |
| 120-125% | 122.5% | (120+125)/2 |
| 125%+ | 125% | Use lower bound |
| N/A | [blank] | Leave empty (formula handles nulls) |
| 110% | 110% | Already valid |

### Fix Procedure

**Option A: Manual fix via browser**
1. Open sheet: https://docs.google.com/spreadsheets/d/1lTpdbDjqW40qe4YUvk_1vBzKYLUNrmLZYyQN-7HmFJg/edit#gid=0
2. Navigate to column Q (NRR)
3. For each range value:
   - Calculate midpoint (e.g., (115+120)/2 = 117.5)
   - Replace with single percentage: `117.5%`
4. For "N/A" → delete content (leave blank)
5. For "125%+" → replace with `125%`

**Option B: Sheets API fix (requires Sheets API enabled)**
```bash
# Enable Sheets API first:
# https://console.developers.google.com/apis/api/sheets.googleapis.com/overview?project=831892255935

# Then use gog CLI:
gog-shapescale --account martin@shapescale.com sheets update \
  1lTpdbDjqW40qe4YUvk_1vBzKYLUNrmLZYyQN-7HmFJg \
  'Equity!Q5' '117.5%'
```

### Impact

Fixing NRR ranges will:
- ✅ Eliminate #VALUE! errors in MSS Score column (Y)
- ✅ Eliminate #VALUE! errors in MSS Rating column (Z)
- ✅ Allow proper numerical analysis and sorting
- ✅ Make formulas copyable to new rows without errors

### Related Columns

Other columns that need single numeric values (not ranges):
- **Column M (Rule of 40 Ops)**: Should be calculated value (Ops Margin + Rev Growth)
- **Column O (Rule of 40 FCF)**: Should be calculated value (FCF Margin + Rev Growth)
- Both can be negative for pre-profitable/turnaround companies

### Prevention

When adding new companies:
1. Always use single percentage values in NRR column
2. Test MSS Score formula immediately after adding row
3. If #VALUE! error appears → check Q column for ranges/text
