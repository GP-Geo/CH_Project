# tests/ → see docs

Testing instructions, fixtures, and conventions are consolidated in
**[../docs/DEVELOPER_GUIDE.md](../docs/DEVELOPER_GUIDE.md#7-testing)**.

```bash
conda run -n ch-heads pytest -q          # full suite
pytest tests/ -v --cov=channel_heads     # with coverage
```
