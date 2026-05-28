# ModelA Data Archive

These archive parts contain generated ModelA data folders only.
The generation programs under `V_Wave_Data_Generate*` are not included.

- Archive parts: 21
- Total archive size: 75.02 GB
- Upload target: Aliyun Drive

## Restore

Extract every `.tar` part from the repository root. Each part contains
`ModelA_Virtual_Internal_Solitary_Wave_Data_Generation/V_Wave_Data*`
relative folder content.

## Checksums

Run this in PowerShell after download:

```powershell
Get-FileHash *.tar -Algorithm SHA256
```

Compare the result with `checksums.sha256`.
