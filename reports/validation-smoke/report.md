# Tournament validation-smoke

Complete: 4/4 matches recorded; 0 pending; 0 colour pairs incomplete. Workers: 2. Engine: 998ba1bb31dc.

League points: win 3, tie 1, loss 0. Forfeits count with the engine outcome; runner errors do not. Compare coverage before ranking a partial run. Intervals describe win proportion, treat ties as non-wins, and assume independent games; repeated maps may violate this assumption. Parallel timing describes a loaded machine and does not prove standalone timing safety.

## Ranking

| Team | Games | Points | W/T/L | Win % | Approx. 95% interval | Mean score | Median | Range | Errors | Own forfeits | Warnings | Timeout events |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 4 | 5 | 1/2/1 | 25.0% | 5%-70% | 0.00 | 0.0 | -28..28 | 0 | 0 | 0 | 0 |
| submission | 4 | 5 | 1/2/1 | 25.0% | 5%-70% | 0.00 | 0.0 | -28..28 | 0 | 0 | 0 | 0 |

## Pairwise W/T/L (row perspective)

| Team vs opponent | baseline | submission |
| --- | --- | --- |
| baseline | - | 1/2/1 |
| submission | 1/2/1 | - |

## Colour and layout breakdown

| Team | Group | Value | Games | Points | W/T/L | Win % | Approx. 95% interval | Mean score | Median | Range | Errors | Own forfeits | Warnings | Timeout events |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | colour | blue | 2 | 1 | 0/1/1 | 0.0% | 0%-66% | -14.00 | -14.0 | -28..0 | 0 | 0 | 0 | 0 |
| baseline | colour | red | 2 | 4 | 1/1/0 | 50.0% | 9%-91% | 14.00 | 14.0 | 0..28 | 0 | 0 | 0 | 0 |
| baseline | layout | RANDOM7 | 2 | 3 | 1/0/1 | 50.0% | 9%-91% | 0.00 | 0.0 | -28..28 | 0 | 0 | 0 | 0 |
| baseline | layout | defaultCapture | 2 | 2 | 0/2/0 | 0.0% | 0%-66% | 0.00 | 0.0 | 0..0 | 0 | 0 | 0 | 0 |
| submission | colour | blue | 2 | 1 | 0/1/1 | 0.0% | 0%-66% | -14.00 | -14.0 | -28..0 | 0 | 0 | 0 | 0 |
| submission | colour | red | 2 | 4 | 1/1/0 | 50.0% | 9%-91% | 14.00 | 14.0 | 0..28 | 0 | 0 | 0 | 0 |
| submission | layout | RANDOM7 | 2 | 3 | 1/0/1 | 50.0% | 9%-91% | 0.00 | 0.0 | -28..28 | 0 | 0 | 0 | 0 |
| submission | layout | defaultCapture | 2 | 2 | 0/2/0 | 0.0% | 0%-66% | 0.00 | 0.0 | 0..0 | 0 | 0 | 0 | 0 |

## Teams and frozen versions

- **baseline**: Provided reflex baseline. Author: Course staff. Approach: Reflex. Components: . Source SHA256: 80143ebae9f80a4edf1b163a4c4c07a8d1f514f5da71e80cef12bb8672ff78ae.

- **submission**: Current root submission. Author: PacLab. Approach: Current submission. Components: . Source SHA256: 6d435e65ec94adb966e835271acf1bc53ef5f196aeeaa012c4a02ac7908a8cc2.

## Match records

| Match | Red | Blue | Layout | Seed | Status | Red score | Seconds | Diagnostic |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | submission | baseline | defaultCapture | 1 | completed | 0 | 0.296 |  |
| 1 | baseline | submission | defaultCapture | 1 | completed | 0 | 0.296 |  |
| 2 | submission | baseline | RANDOM7 | 2 | completed | 28 | 0.35 |  |
| 3 | baseline | submission | RANDOM7 | 2 | completed | 28 | 0.35 |  |

Exact configuration, versions and schedule: `manifest.json`. Raw results: `matches.jsonl`. Detailed logs and optional replays remain under the local run directory; they are not included in this shared report.
