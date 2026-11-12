"""Explicit rerun namespaces; frozen upstream inputs remain separate."""
import os
from pathlib import Path

ARTIFACT_ROOT=Path(os.environ.get('RECSYS_CAMPAIGN_ROOT','checkpoints/dynamic-campaign-v1'))
EVIDENCE_ROOT=Path(os.environ.get('RECSYS_CAMPAIGN_EVIDENCE','experiments/dynamic_campaign'))
SOURCE_EVIDENCE=Path('experiments/dynamic_campaign')
