# S184: recover benchmark headroom without losing evidence

Recovered 4,659,016,177 logical bytes of disk space by replacing 24 completed
Nsight scratch files with verified lossless ZIPs. C: increased from about
1.08 GB free to 5.74 GB. All 6,578 inventoried artifacts across the affected
stages and S183's 91-file inventory remain unchanged. This restores room for
a fresh encoder qualification campaign; it does not establish the cause of
S182's earlier `ENOSPC` error. Production remains unchanged.

## Scope and discovery

The parent is `cd8a3ebd6f253863dda17c390d6065eb338d90ab` on `feat/cuda`.
[S183](cuda-reverse-bit-packing-s183.md) has passed its primitive and integrated
host tests, but still needs whole-image and performance qualification.
[S182](cuda-bounded-appender-qualification-s182.md) remains an interrupted
campaign and is not resumed or reclassified here.

At the start of this turn, C: had 1,081,839,616 bytes free and U: 30,883,840.
No measured executable was running. S182/S183 temporary directories were
empty. A read-only walk of task-stage temporary directories, skipping reparse
points, found 77 files totaling 4,664,082,300 bytes. The selected 24 ETLs
account for 4,664,066,048 bytes; the other 53 tiny scratch files were left
alone. No personal files, protected notes, source files, frozen artifacts,
global caches, system folders or recovery-volume contents were removed.

The exact selected filenames in each listed stage were:

```text
temp/nvidia/nsight_systems/etl_files/dxg_krnl_profiler_session.etl
temp/nvidia/nsight_systems/etl_files/Merged.etl
temp/nvidia/nsight_systems/etl_files/reflex_stats_session_profiler_session.etl
```

Every selected path is excluded from its stage's frozen inventory. All
6,578 inventoried files were rehashed before recovery, not merely assumed
unchanged because their manifests existed. The same complete inventory
check passed after recovery.

## Preservation before removal

`s184_archive_etl.ps1` accepts only the eight explicit stage numbers below.
It resolves each target under that stage's task-local ETL directory, checks
every ancestor for reparse points, rejects any inventoried target, and
requires no active diagnostic/compiler process. It never recursively deletes
a directory or enumerates paths in one shell for deletion in another.

Each stage needs enough free space for its entire uncompressed input size
plus 128 MiB before it starts. Outputs use exclusive creation. Before
compression, a manifest records all three source lengths and SHA-256 hashes.
After compression, every ZIP entry is decompressed as a stream and checked
against that length/hash; each original is rehashed as well. The frozen
manifest hash must remain unchanged. A separate verified-archive record is
written before removal. Only then are the three exact originals removed
with PowerShell `Remove-Item -LiteralPath`, and a removal ledger is written.

Archives and ledgers are under
`build-cuda-ninja/profiles/s184-artifacts/recovered_scratch`, outside the
older frozen roots. They preserve every original byte, not just exported
trace summaries. No ZIP entry is discarded or rewritten during verification.

| Stage | Unchanged inventory files | Removed raw bytes | ZIP bytes |
|---|---:|---:|---:|
| S139 | 822 | 587,202,560 | 635,516 |
| S140 | 511 | 587,202,560 | 634,792 |
| S141 | 548 | 587,202,560 | 631,023 |
| S142 | 738 | 587,202,560 | 635,360 |
| S143 | 1,604 | 587,202,560 | 633,268 |
| S144 | 457 | 587,202,560 | 644,571 |
| S153 | 447 | 553,648,128 | 600,965 |
| S154 | 1,451 | 587,202,560 | 634,376 |
| Total | 6,578 | 4,664,066,048 | 5,049,871 |

The net byte calculation excludes small manifests/logs and filesystem
overhead. Observed volume free-space changes can also include unrelated
system activity; they are not asserted equal to that logical calculation.
The final removal ledger records 5,740,507,136 bytes free on C:, followed by
a volume query reporting 5,740,298,240 bytes. U: and the mounted Windows
Recovery volume were not used as destinations.

## Launcher failure and recovery

The first archive job ran in Windows PowerShell 5.1 and failed to resolve
`Get-FileHash` before creating an archive or deleting any file. Its original
log/journal are preserved. A subsequent ordinary shell probe resolved that
function, so this evidence does not establish a general PowerShell 5.1
limitation, a missing installation, or a privilege cause.

The new `s184_resume_pwsh.py` pins and invokes the installed PowerShell 7.6.5
executable explicitly. A preflight resolves the required commands and hashes
the archive script under the same runner environment. It then invokes the
unchanged pinned archive script. No module-path, installation, permission,
execution-policy setting, firewall or other machine setting is modified.
The failed job is not overwritten or reported as successful.

All eight archive jobs passed on 2026-09-10, from 10:14:03.697 to
10:14:35.875 UTC, after the PowerShell 7 preflight. There are ten journals:
nine accepted and one preserved pre-archive rejection. No benchmark, CUDA
sanitizer or compiler ran during compression. This work has no performance
measurement or codec-correctness claim of its own.

## Verification and next step

The verifier rehashes every older inventoried artifact, S183's inventory and
pinned sources; checks all job commands, logs and nonoverlap; verifies all
ZIPs and every decompressed original SHA; and confirms that only the exact
24 non-inventoried paths are absent. Each stage's before/verified/removal
ledgers and all recovery helpers are frozen with the new evidence.

Read-only verification from the repository root:

```powershell
python build-cuda-ninja/profiles/s184_verify.py --frozen
```

The removed files are recoverable from the eight `sNNN_etl.zip` archives;
their original absolute destinations, entry names, lengths and hashes are
recorded in the corresponding ledgers. Restoration is not performed here
because it would consume the reclaimed headroom.

The storage-headroom obstacle is resolved for the next bounded campaign.
S182's disk-error cause remains unproven, and its incomplete timing evidence
stays incomplete. Proceed with a new independently pinned S183 whole-encoder
qualification and performance experiment, retaining failure information if
storage errors recur. No production change is promoted, and the optimization
goal remains active rather than declared maxed out.
